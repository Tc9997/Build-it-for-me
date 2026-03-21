"""Template cache: local, offline-safe, keyed by template_id + pinned_commit.

On cache hit, returns the cached path without any network or I/O beyond
the local filesystem. On cache miss, copies from source to cache.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from build_loop.templates.models import RegistryEntry
from build_loop.templates.registry import _compute_content_hash


class CacheError(Exception):
    """Raised for cache failures."""


_DEFAULT_CACHE_DIR = Path.home() / ".build-loop" / "template-cache"


def cache_key(entry: RegistryEntry) -> str:
    """Canonical cache key for a registry entry."""
    return f"{entry.template_id}-{entry.pinned_commit[:16]}"


def get_cached(entry: RegistryEntry, cache_dir: Path | None = None) -> Path | None:
    """Check for a cache hit. Returns the cached template path or None.

    Offline-safe: no network, no subprocess, just a directory check
    plus a content hash verification.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cached = cache_dir / cache_key(entry)
    if not cached.is_dir():
        return None

    # Verify content hash matches — detect corruption/tampering
    actual_hash = _compute_content_hash(cached)
    if actual_hash != entry.pinned_commit:
        return None

    return cached


def populate(entry: RegistryEntry, cache_dir: Path | None = None) -> Path:
    """Populate the cache from the template source. Returns the cache path.

    For local sources: copies the directory.
    Raises CacheError if the source is missing or the hash doesn't match after copy.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = cache_key(entry)
    target = cache_dir / key

    # Remove stale cache entry if exists
    if target.exists():
        shutil.rmtree(target)

    source = Path(entry.source_path)
    if not source.is_dir():
        raise CacheError(f"Template source not found: {source}")

    shutil.copytree(source, target)

    # Verify the copy matches
    actual_hash = _compute_content_hash(target)
    if actual_hash != entry.pinned_commit:
        shutil.rmtree(target)
        raise CacheError(
            f"Cache population failed: content hash mismatch after copy. "
            f"Expected {entry.pinned_commit[:16]}, got {actual_hash[:16]}"
        )

    return target


def ensure_cached(entry: RegistryEntry, cache_dir: Path | None = None) -> Path:
    """Get from cache or populate. Returns the cache path."""
    cached = get_cached(entry, cache_dir)
    if cached is not None:
        return cached
    return populate(entry, cache_dir)
