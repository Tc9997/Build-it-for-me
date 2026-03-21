"""Template registry: maps archetypes to pinned template entries.

Local-only, deterministic, two archetypes. Rejects anything else.

Template pins are immutable expected hashes checked into pinned_hashes.json.
At import time, the live fixture content is verified against these expected
hashes. If the content has drifted, the registry refuses to load that
template — a changed fixture does NOT become its own new pin.

To update a pin after intentionally changing a fixture:
  python -m build_loop.templates.registry --update-pins
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from build_loop.templates.models import Archetype, RegistryEntry


class RegistryError(Exception):
    """Raised for registry lookup failures."""


def _fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def _pinned_hashes_path() -> Path:
    return Path(__file__).parent / "pinned_hashes.json"


def _load_pinned_hashes() -> dict[str, str]:
    """Load the checked-in expected hashes. These are the immutable pins."""
    path = _pinned_hashes_path()
    if not path.exists():
        raise RegistryError(f"Pinned hashes file missing: {path}")
    return json.loads(path.read_text())


def _compute_content_hash(directory: Path) -> str:
    """SHA-256 of all file contents in a directory, sorted by path."""
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(directory))
            h.update(rel.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Build the registry at import time, verifying against checked-in pins
# ---------------------------------------------------------------------------

def _build_registry() -> dict[Archetype, RegistryEntry]:
    """Build the registry from bundled fixtures, verifying against pinned hashes."""
    fixtures = _fixtures_dir()
    pinned = _load_pinned_hashes()
    entries = {}

    for archetype in (Archetype.PYTHON_CLI, Archetype.FASTAPI_SERVICE):
        template_id = f"{archetype.value}_v1"
        template_dir = fixtures / archetype.value

        if not template_dir.is_dir():
            continue

        expected_hash = pinned.get(template_id)
        if not expected_hash:
            raise RegistryError(
                f"No pinned hash for template '{template_id}' in pinned_hashes.json. "
                f"Run: python -m build_loop.templates.registry --update-pins"
            )

        actual_hash = _compute_content_hash(template_dir)
        if actual_hash != expected_hash:
            raise RegistryError(
                f"Template '{template_id}' content has drifted from pinned hash. "
                f"Expected: {expected_hash[:16]}..., actual: {actual_hash[:16]}... "
                f"If intentional, run: python -m build_loop.templates.registry --update-pins"
            )

        entries[archetype] = RegistryEntry(
            template_id=template_id,
            archetype=archetype,
            source_type="local",
            source_path=str(template_dir),
            pinned_commit=expected_hash,
        )

    return entries


_REGISTRY: dict[Archetype, RegistryEntry] | None = None


def _get_registry() -> dict[Archetype, RegistryEntry]:
    """Lazy initialization — registry is built on first access, not at import."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


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

    entry = _get_registry().get(archetype)
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
    return [a.value for a in _get_registry().keys()]


# ---------------------------------------------------------------------------
# CLI for updating pins
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if "--update-pins" in sys.argv:
        fixtures = _fixtures_dir()
        pins = {}
        for archetype in (Archetype.PYTHON_CLI, Archetype.FASTAPI_SERVICE):
            template_dir = fixtures / archetype.value
            if template_dir.is_dir():
                template_id = f"{archetype.value}_v1"
                h = _compute_content_hash(template_dir)
                pins[template_id] = h
                print(f"  {template_id}: {h}")
        _pinned_hashes_path().write_text(json.dumps(pins, indent=2) + "\n")
        print(f"Updated {_pinned_hashes_path()}")
    else:
        print("Usage: python -m build_loop.templates.registry --update-pins")
