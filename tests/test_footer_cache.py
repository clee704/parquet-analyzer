"""Tests for the persistent footer cache (:mod:`parquet_analyzer._footer_cache`)."""

from __future__ import annotations

import io
import os
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_analyzer import _footer_cache
from parquet_analyzer.parquet_file import ParquetFile


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    """Isolate the cache in a tmp dir and ensure it is enabled. Returns the
    footer-cache subdirectory path (created on first use)."""
    base = tmp_path / "cache"
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(base))
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    monkeypatch.delenv("PARQUET_ANALYZER_CACHE_MAX_BYTES", raising=False)
    return base / "footer"


@pytest.fixture()
def always_cache(monkeypatch):
    """Lower the worth-caching threshold so small test files are cached."""
    monkeypatch.setattr(_footer_cache, "CHUNK_THRESHOLD", 0)


def _write_parquet(path, *, ncols=3, nrows=30, row_group_size=10):
    cols = {
        f"c{i}": pa.array([float(i + j) for j in range(nrows)]) for i in range(ncols)
    }
    pq.write_table(
        pa.table(cols), path, row_group_size=row_group_size, write_statistics=True
    )
    return path


# ---------------------------------------------------------------------------
# End-to-end: round-trip and equivalence
# ---------------------------------------------------------------------------


def test_cache_round_trip_hit(tmp_path, cache_dir, always_cache):
    p = _write_parquet(tmp_path / "f.parquet")
    with ParquetFile(str(p)) as pf1:
        cached_footer_offset = pf1._footer_offset
    files = list(cache_dir.glob("*.pkl"))
    assert len(files) == 1, "first open should write exactly one cache entry"

    with ParquetFile(str(p)) as pf2:
        assert pf2._footer_offset == cached_footer_offset
        assert len(pf2._footer_thrift.row_groups) == 3


def test_cache_hit_does_not_reparse(tmp_path, cache_dir, always_cache, monkeypatch):
    p = _write_parquet(tmp_path / "f.parquet")
    with ParquetFile(str(p)):  # warm the cache
        pass
    # On the next open, _parse_footer must NOT be called (served from cache).
    import parquet_analyzer.parquet_file as pf_mod

    calls = {"n": 0}
    original = pf_mod._parse_footer

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    monkeypatch.setattr(pf_mod, "_parse_footer", counting)
    with ParquetFile(str(p)):
        pass
    assert calls["n"] == 0, "cache hit must skip _parse_footer"


def test_cache_output_equivalent_to_fresh_parse(tmp_path, cache_dir, always_cache):
    p = _write_parquet(tmp_path / "f.parquet")
    with ParquetFile(str(p), use_cache=False) as pf_fresh:
        fresh = pf_fresh.to_json(view="tree", depth="all")
    with ParquetFile(str(p)):  # warm
        pass
    with ParquetFile(str(p)) as pf_cached:
        cached = pf_cached.to_json(view="tree", depth="all")
    assert cached == fresh

    with ParquetFile(str(p), use_cache=False) as a:
        fresh_layout = a.to_json(view="layout", depth="all")
    with ParquetFile(str(p)) as b:
        cached_layout = b.to_json(view="layout", depth="all")
    assert cached_layout == fresh_layout


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_modified_file_misses_cache(tmp_path, cache_dir, always_cache):
    """Content-addressed key: rewriting the file with a different footer must
    produce a fresh (correct) parse, never a stale hit."""
    p = tmp_path / "f.parquet"
    _write_parquet(p, ncols=3)
    with ParquetFile(str(p)) as pf:
        assert len(pf._footer_thrift.row_groups[0].columns) == 3
    # Rewrite with a different schema -> different footer -> different key.
    _write_parquet(p, ncols=5)
    with ParquetFile(str(p)) as pf:
        assert len(pf._footer_thrift.row_groups[0].columns) == 5


def test_corrupt_cache_entry_falls_back(tmp_path, cache_dir, always_cache):
    p = _write_parquet(tmp_path / "f.parquet")
    with ParquetFile(str(p)):  # warm
        pass
    entry = next(cache_dir.glob("*.pkl"))
    entry.write_bytes(b"not a pickle")
    # Must not raise; falls back to a real parse.
    with ParquetFile(str(p)) as pf:
        assert len(pf._footer_thrift.row_groups) == 3


def test_key_mismatch_in_payload_is_ignored(tmp_path, cache_dir, always_cache):
    p = _write_parquet(tmp_path / "f.parquet")
    with ParquetFile(str(p)):  # warm
        pass
    entry = next(cache_dir.glob("*.pkl"))
    # Tamper the embedded key so it no longer matches the filename's key.
    payload = pickle.loads(entry.read_bytes())
    payload["key"] = "deadbeef"
    entry.write_bytes(pickle.dumps(payload))
    with ParquetFile(str(p)) as pf:  # ignored -> reparse, no crash
        assert len(pf._footer_thrift.row_groups) == 3


# ---------------------------------------------------------------------------
# Threshold / opt-out
# ---------------------------------------------------------------------------


def test_small_footer_not_cached(tmp_path, cache_dir):
    """With the real threshold, a tiny footer is not cached."""
    p = _write_parquet(tmp_path / "f.parquet", ncols=2, nrows=4, row_group_size=4)
    with ParquetFile(str(p)):
        pass
    assert not cache_dir.exists() or not list(cache_dir.glob("*.pkl"))


def test_use_cache_false_bypasses(tmp_path, cache_dir, always_cache):
    p = _write_parquet(tmp_path / "f.parquet")
    with ParquetFile(str(p), use_cache=False):
        pass
    assert not cache_dir.exists() or not list(cache_dir.glob("*.pkl"))


def test_no_cache_env_disables(tmp_path, cache_dir, always_cache, monkeypatch):
    monkeypatch.setenv("PARQUET_ANALYZER_NO_CACHE", "1")
    assert not _footer_cache.enabled()
    p = _write_parquet(tmp_path / "f.parquet")
    with ParquetFile(str(p)):
        pass
    assert not cache_dir.exists() or not list(cache_dir.glob("*.pkl"))


# ---------------------------------------------------------------------------
# Unit-level: keying, dir trust, eviction
# ---------------------------------------------------------------------------


def test_read_footer_bytes_guards():
    assert _footer_cache.read_footer_bytes(io.BytesIO(b"short"), 5) is None
    # Implausible footer_size (claims a huge footer): rejected.
    buf = io.BytesIO(b"\x00" * 4 + b"\xff\xff\xff\xff" + b"PAR1")
    assert _footer_cache.read_footer_bytes(buf, 12) is None
    # Truncated trailer -> struct.error path -> None.
    assert _footer_cache.read_footer_bytes(io.BytesIO(b"ab"), 20) is None


def test_read_footer_bytes_valid():
    import struct

    footer = b"FOOTERBYTES"
    blob = b"PAR1" + footer + struct.pack("<I", len(footer)) + b"PAR1"
    size = len(blob)
    got = _footer_cache.read_footer_bytes(io.BytesIO(blob), size)
    assert got == footer


def test_compute_key_deterministic_and_sensitive():
    k1 = _footer_cache.compute_key(100, b"abc")
    assert k1 == _footer_cache.compute_key(100, b"abc")
    assert k1 != _footer_cache.compute_key(101, b"abc")
    assert k1 != _footer_cache.compute_key(100, b"abd")


def test_enabled_toggle(monkeypatch):
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    assert _footer_cache.enabled()
    monkeypatch.setenv("PARQUET_ANALYZER_NO_CACHE", "1")
    assert not _footer_cache.enabled()
    monkeypatch.setenv("PARQUET_ANALYZER_NO_CACHE", "true")
    assert not _footer_cache.enabled()


def test_max_cache_bytes_env(monkeypatch):
    monkeypatch.delenv("PARQUET_ANALYZER_CACHE_MAX_BYTES", raising=False)
    assert _footer_cache.max_cache_bytes() == _footer_cache._DEFAULT_MAX_CACHE_BYTES
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_MAX_BYTES", "4096")
    assert _footer_cache.max_cache_bytes() == 4096
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_MAX_BYTES", "notanint")
    assert _footer_cache.max_cache_bytes() == _footer_cache._DEFAULT_MAX_CACHE_BYTES


def test_cache_dir_disabled_when_no_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("PARQUET_ANALYZER_NO_CACHE", "1")
    assert _footer_cache.cache_dir() is None


def test_cache_dir_untrusted_when_world_writable(monkeypatch, tmp_path):
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX permission semantics only")
    base = tmp_path / "c"
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(base))
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    footer = base / "footer"
    footer.mkdir(parents=True)
    os.chmod(footer, 0o777)  # group/world-writable -> untrusted
    assert _footer_cache.cache_dir() is None


def test_cache_dir_trusted(monkeypatch, tmp_path):
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    d = _footer_cache.cache_dir()
    assert d is not None and d.is_dir()


def test_base_cache_dir_resolution(monkeypatch, tmp_path):
    # Explicit override
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(tmp_path / "ov"))
    assert _footer_cache._base_cache_dir() == tmp_path / "ov"
    # XDG (posix)
    monkeypatch.delenv("PARQUET_ANALYZER_CACHE_DIR", raising=False)
    monkeypatch.setattr(_footer_cache.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert _footer_cache._base_cache_dir() == tmp_path / "xdg" / "parquet-analyzer"
    # default ~/.cache
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(
        _footer_cache.Path, "home", staticmethod(lambda: tmp_path / "h")
    )
    assert (
        _footer_cache._base_cache_dir()
        == tmp_path / "h" / ".cache" / "parquet-analyzer"
    )


def test_base_cache_dir_windows(monkeypatch, tmp_path):
    monkeypatch.delenv("PARQUET_ANALYZER_CACHE_DIR", raising=False)
    monkeypatch.setattr(_footer_cache.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    got = _footer_cache._base_cache_dir()
    assert got == tmp_path / "local" / "parquet-analyzer" / "Cache"
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(
        _footer_cache.Path, "home", staticmethod(lambda: tmp_path / "h")
    )
    got = _footer_cache._base_cache_dir()
    assert got.parts[-3:] == ("Local", "parquet-analyzer", "Cache")


def test_dir_is_trusted_nonexistent_and_nondir(tmp_path):
    assert not _footer_cache._dir_is_trusted(tmp_path / "nope")
    f = tmp_path / "afile"
    f.write_text("x")
    assert not _footer_cache._dir_is_trusted(f)


def test_load_miss_and_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    assert _footer_cache.load("nonexistentkey") is None  # miss
    monkeypatch.setenv("PARQUET_ANALYZER_NO_CACHE", "1")
    assert _footer_cache.load("anykey") is None  # disabled


def test_store_below_threshold_and_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    parsed = (object(), {}, 0, {}, {})
    _footer_cache.store("k", parsed, chunk_count=5)  # below default threshold
    d = tmp_path / "c" / "footer"
    assert not d.exists() or not list(d.glob("*.pkl"))


def test_store_load_unit_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    monkeypatch.setattr(_footer_cache, "CHUNK_THRESHOLD", 0)
    parsed = ("thrift", {"name": "footer"}, 123, {"h": 1}, {"t": 2})
    _footer_cache.store("mykey", parsed, chunk_count=0)
    got = _footer_cache.load("mykey")
    assert got == parsed


def test_evict_if_needed(tmp_path):
    d = tmp_path / "footer"
    d.mkdir()
    # Three 100-byte entries; ceiling 250 -> oldest one evicted.
    import time

    paths = []
    for i in range(3):
        p = d / f"e{i}.pkl"
        p.write_bytes(b"x" * 100)
        os.utime(p, (1000 + i, 1000 + i))  # ascending mtime: e0 oldest
        paths.append(p)
    _footer_cache._evict_if_needed(d, ceiling=250)
    remaining = {p.name for p in d.glob("*.pkl")}
    assert "e0.pkl" not in remaining
    assert {"e1.pkl", "e2.pkl"} <= remaining


def test_evict_noop_under_ceiling(tmp_path):
    d = tmp_path / "footer"
    d.mkdir()
    (d / "a.pkl").write_bytes(b"x" * 10)
    _footer_cache._evict_if_needed(d, ceiling=10_000)
    assert (d / "a.pkl").exists()


def test_column_chunk_count():
    class _RG:
        def __init__(self, n):
            self.columns = list(range(n))

    class _Footer:
        row_groups = [_RG(3), _RG(3), _RG(4)]

    assert _footer_cache.column_chunk_count(_Footer()) == 10


# ---------------------------------------------------------------------------
# Defensive / error paths
# ---------------------------------------------------------------------------


def test_abi_signature_handles_missing_thrift_metadata(monkeypatch):
    def boom(_name):
        raise RuntimeError("no metadata")

    monkeypatch.setattr("importlib.metadata.version", boom)
    # Should not raise; just folds in "unknown" for the thrift version.
    assert _footer_cache.compute_key(10, b"x")  # non-empty hex digest


def test_resolve_cache_dir_mkdir_failure(monkeypatch, tmp_path):
    # Point the cache base at a regular file so mkdir(.../footer) fails.
    afile = tmp_path / "afile"
    afile.write_text("x")
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(afile))
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    assert _footer_cache.cache_dir() is None


def test_dir_is_trusted_uid_mismatch(monkeypatch, tmp_path):
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX only")
    d = tmp_path / "footer"
    d.mkdir(mode=0o700)
    other_uid = os.getuid() + 12345
    monkeypatch.setattr(_footer_cache.os, "getuid", lambda: other_uid)
    assert not _footer_cache._dir_is_trusted(d)


def test_load_unpickle_surprise_is_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    d = _footer_cache.cache_dir()
    (d / "k.pkl").write_bytes(b"whatever")
    monkeypatch.setattr(
        _footer_cache.pickle,
        "load",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _footer_cache.load("k") is None


def test_store_noop_when_dir_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("PARQUET_ANALYZER_NO_CACHE", "1")
    monkeypatch.setattr(_footer_cache, "CHUNK_THRESHOLD", 0)
    # Threshold passes, but cache_dir() is None (disabled) -> no write, no error.
    _footer_cache.store("k", ("t", {}, 0, {}, {}), chunk_count=5)
    assert not (tmp_path / "c" / "footer").exists()


def test_store_write_failure_is_swallowed(monkeypatch, tmp_path):
    monkeypatch.setenv("PARQUET_ANALYZER_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.delenv("PARQUET_ANALYZER_NO_CACHE", raising=False)
    monkeypatch.setattr(_footer_cache, "CHUNK_THRESHOLD", 0)

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(_footer_cache.os, "replace", boom)
    # Must not raise; the temp file is cleaned up and no entry is left.
    _footer_cache.store("k", ("t", {}, 0, {}, {}), chunk_count=0)
    d = tmp_path / "c" / "footer"
    assert not list(d.glob("*.pkl"))
    assert not list(d.glob("*.tmp"))


def test_silent_unlink_missing_path(tmp_path):
    # Must not raise on a nonexistent path.
    _footer_cache._silent_unlink(tmp_path / "nope" / "x")


def test_evict_skips_unstatable_entry(tmp_path):
    d = tmp_path / "footer"
    d.mkdir()
    (d / "real.pkl").write_bytes(b"x" * 10)
    dangling = d / "dangling.pkl"
    os.symlink(tmp_path / "does-not-exist", dangling)  # stat() will fail
    # Should not raise; the unstatable entry is skipped.
    _footer_cache._evict_if_needed(d, ceiling=0)
