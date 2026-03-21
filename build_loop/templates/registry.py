"""Template registry: maps archetypes to pinned template entries.

Local-only, deterministic, two archetypes. Rejects anything else.
Entries are content-pinned via SHA-256 of the template directory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from build_loop.templates.models import Archetype, RegistryEntry


class RegistryError(Exception):
    """Raised for registry lookup failures."""


def _fixtures_dir() -> Path:
    """Path to the bundled template fixtures."""
    return Path(__file__).parent / "fixtures"


def _compute_content_hash(directory: Path) -> str:
    """SHA-256 of all file contents in a directory, sorted by path.

    Deterministic: same files in same order produce the same hash.
    """
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(directory))
            h.update(rel.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Build the registry at import time (two entries, deterministic)
# ---------------------------------------------------------------------------

def _build_registry() -> dict[Archetype, RegistryEntry]:
    """Build the registry from bundled fixtures."""
    fixtures = _fixtures_dir()
    entries = {}

    for archetype in (Archetype.PYTHON_CLI, Archetype.FASTAPI_SERVICE):
        template_dir = fixtures / archetype.value
        if not template_dir.is_dir():
            continue
        entries[archetype] = RegistryEntry(
            template_id=f"{archetype.value}_v1",
            archetype=archetype,
            source_type="local",
            source_path=str(template_dir),
            pinned_commit=_compute_content_hash(template_dir),
        )

    return entries


_REGISTRY = _build_registry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve(archetype: str | Archetype) -> RegistryEntry:
    """Resolve an archetype to a registry entry.

    Raises RegistryError if the archetype is not supported or not found.
    """
    if isinstance(archetype, str):
        try:
            archetype = Archetype(archetype)
        except ValueError:
            raise RegistryError(
                f"Unsupported archetype: {archetype!r}. "
                f"Supported: {[a.value for a in Archetype]}"
            )

    entry = _REGISTRY.get(archetype)
    if entry is None:
        raise RegistryError(f"No template registered for archetype: {archetype.value}")

    return entry


def verify_commit(entry: RegistryEntry) -> bool:
    """Verify that the template source still matches the pinned commit hash."""
    source = Path(entry.source_path)
    if not source.is_dir():
        return False
    actual = _compute_content_hash(source)
    return actual == entry.pinned_commit


def list_archetypes() -> list[str]:
    """Return the list of supported archetype names."""
    return [a.value for a in _REGISTRY.keys()]
