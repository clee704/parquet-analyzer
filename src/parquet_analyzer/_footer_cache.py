"""Persistent, content-addressed cache for the parsed parquet footer.

The CLI is one-shot: every command (`show FILE PATH`, `tree`, `layout`, …)
is a fresh process that re-opens the file and re-parses its footer. For a
complex footer (hundreds of row groups × tens of columns) the
offset-recording thrift decode dominates — seconds per open — so a 15-step
interactive exploration would re-pay it 15 times.

This module caches the parsed footer on disk so repeated opens of an
unchanged file skip the decode.

Design:

- **Content-addressed key.** The key is ``sha256(abi || file_size ||
  footer_bytes)``. Hashing the footer bytes (cheap — they must be read to
  parse anyway) makes the cache *self-invalidating*: any change to the
  footer changes the key, so a stale entry can never be served. It also
  makes the cache reusable across symlinks / copies / different working
  directories (same footer ⇒ same key ⇒ hit), and the embedded ABI
  signature invalidates the whole cache automatically after a thrift or
  tool upgrade (a changed ABI ⇒ different keys ⇒ old entries are never
  looked up).

- **Directory trust is the security boundary.** The payload is a pickle, so
  the cache directory must be private: created ``0700``, and on reuse
  verified to be a directory owned by the current user with no group/world
  access. If that check fails the cache is silently disabled (parsing still
  works). Entries this process writes are ``0600``.

- **Write only when it pays off.** A footer is cached only when it has at
  least :data:`CHUNK_THRESHOLD` column chunks — small footers parse fast
  enough that the write overhead and disk clutter aren't worth it.

- **Bounded size.** On write, if the cache directory exceeds
  :data:`max_cache_bytes`, the oldest entries are evicted until it fits.

The cache is transparent (commands stay stateless) and fully optional:
set ``PARQUET_ANALYZER_NO_CACHE=1`` (or pass ``use_cache=False`` to
:class:`~parquet_analyzer.ParquetFile`) to bypass it, and
``PARQUET_ANALYZER_CACHE_DIR`` to relocate it.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import stat
import struct
import sys
from pathlib import Path
from typing import Any

# Bump on any change to the cached representation (the 5-tuple shape, the
# segment dict layout, etc.). A bump changes the ABI signature, so every
# key changes and old entries are never read.
CACHE_FORMAT_VERSION = 1

# A footer is cached only when it has at least this many column chunks —
# the deterministic proxy for "the offset-recording decode was expensive
# enough that caching pays off". Smaller footers parse fast; caching them
# only adds write overhead and clutter.
CHUNK_THRESHOLD = 1000

_DEFAULT_MAX_CACHE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB

_NO_CACHE_ENV = "PARQUET_ANALYZER_NO_CACHE"
_CACHE_DIR_ENV = "PARQUET_ANALYZER_CACHE_DIR"
_MAX_BYTES_ENV = "PARQUET_ANALYZER_CACHE_MAX_BYTES"

# Parsed-footer 5-tuple, as returned by ``_core._parse_footer``.
ParsedFooter = tuple


def enabled() -> bool:
    """Whether the cache is enabled (not switched off via the environment)."""
    return os.environ.get(_NO_CACHE_ENV, "") not in ("1", "true", "True")


def max_cache_bytes() -> int:
    """Eviction ceiling for the cache directory (overridable via
    ``PARQUET_ANALYZER_CACHE_MAX_BYTES``); falls back to the default on an
    unparseable value."""
    raw = os.environ.get(_MAX_BYTES_ENV)
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return _DEFAULT_MAX_CACHE_BYTES


def _abi_signature() -> bytes:
    """Identity of everything that affects the cached representation. Folded
    into the key so an upgrade can never serve an incompatible entry."""
    from . import __version__

    try:
        import importlib.metadata as _md

        thrift_version = _md.version("thrift")
    except Exception:
        thrift_version = "unknown"
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    sig = f"v{CACHE_FORMAT_VERSION}|pa={__version__}|py={py}|thrift={thrift_version}"
    return sig.encode("utf-8")


def read_footer_bytes(f, file_size: int) -> bytes | None:
    """Read just the footer thrift bytes — cheaply, for keying — or return
    ``None`` if the file is too small or its footer length is implausible
    (in which case the caller skips the cache and lets the real parse raise
    the canonical error).
    """
    if file_size < 12:  # 4-byte header + 8-byte trailer at minimum
        return None
    try:
        f.seek(file_size - 8)
        footer_size = struct.unpack("<I", f.read(4))[0]
        if footer_size <= 0 or footer_size > file_size - 8:
            return None
        f.seek(file_size - 8 - footer_size)
        return f.read(footer_size)
    except (OSError, struct.error):
        return None


def compute_key(file_size: int, footer_bytes: bytes) -> str:
    """Content-addressed cache key for a footer."""
    h = hashlib.sha256()
    h.update(_abi_signature())
    h.update(struct.pack("<Q", file_size))
    h.update(footer_bytes)
    return h.hexdigest()


def cache_dir() -> Path | None:
    """Resolve, create (``0700``), and trust-check the cache directory.

    Returns the directory, or ``None`` when the cache is disabled or the
    directory cannot be made/confirmed private.
    """
    if not enabled():
        return None
    return _resolve_cache_dir()


def _base_cache_dir() -> Path:
    override = os.environ.get(_CACHE_DIR_ENV)
    if override:
        return Path(override)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "parquet-analyzer" / "Cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "parquet-analyzer"


def _resolve_cache_dir() -> Path | None:
    d = _base_cache_dir() / "footer"
    try:
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        return None
    if not _dir_is_trusted(d):
        return None
    return d


def _dir_is_trusted(d: Path) -> bool:
    """A trusted cache dir is a real directory, owned by us, with no
    group/world access (POSIX). On platforms without uid/permission
    semantics the per-user base directory is the boundary."""
    try:
        st = os.stat(d)
    except OSError:
        return False
    if not stat.S_ISDIR(st.st_mode):
        return False
    if hasattr(os, "getuid"):
        if st.st_uid != os.getuid():
            return False
        if st.st_mode & 0o077:
            return False
    return True


def load(key: str) -> ParsedFooter | None:
    """Return the cached parsed-footer 5-tuple for ``key``, or ``None`` on a
    miss or any read/unpickle/verification failure (the caller then parses
    normally)."""
    d = cache_dir()
    if d is None:
        return None
    path = d / f"{key}.pkl"
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        # Any failure — missing file, truncated/corrupt pickle, a class
        # moved across versions — is treated as a miss so the real parse
        # always proceeds. The cache must never break correctness.
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("key") != key
        or not isinstance(payload.get("parsed"), tuple)
        or len(payload["parsed"]) != 5
    ):
        return None
    return payload["parsed"]


def store(key: str, parsed: ParsedFooter, chunk_count: int) -> None:
    """Persist a parsed footer if it is worth caching. No-op when the cache
    is disabled/untrusted or the footer is below :data:`CHUNK_THRESHOLD`
    column chunks. Best-effort: any write error is swallowed."""
    if chunk_count < CHUNK_THRESHOLD:
        return
    d = cache_dir()
    if d is None:
        return
    payload = {"key": key, "parsed": parsed}
    path = d / f"{key}.pkl"
    tmp = d / f"{key}.{os.getpid()}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            pass
        os.replace(tmp, path)
    except OSError:
        _silent_unlink(tmp)
        return
    _evict_if_needed(d, max_cache_bytes())


def _silent_unlink(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _evict_if_needed(d: Path, ceiling: int) -> None:
    """Delete the oldest entries (by mtime) until the directory's total
    ``.pkl`` size is within ``ceiling``. Best-effort."""
    try:
        entries = []
        total = 0
        for p in d.glob("*.pkl"):
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        if total <= ceiling:
            return
        entries.sort()  # oldest first
        for _mtime, size, p in entries:
            if total <= ceiling:
                break
            _silent_unlink(p)
            total -= size
    except OSError:
        return


def column_chunk_count(footer_thrift: Any) -> int:
    """Total column chunks in a footer — the threshold metric for
    :func:`store`."""
    return sum(len(rg.columns) for rg in footer_thrift.row_groups)
