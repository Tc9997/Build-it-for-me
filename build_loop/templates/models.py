"""Template models: archetype, registry entry, ownership manifest.

Schema version is explicit on all persisted models.
extra="forbid" on all strict models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Archetype — the only two supported project types
# ---------------------------------------------------------------------------

class Archetype(str, Enum):
    PYTHON_CLI = "python_cli"
    FASTAPI_SERVICE = "fastapi_service"


# ---------------------------------------------------------------------------
# File ownership categories
# ---------------------------------------------------------------------------

class FileOwner(str, Enum):
    TEMPLATE_LOCKED = "template_locked"  # From template, do not modify
    BUILDER_OWNED = "builder_owned"      # Template slot — builder fills domain logic
    GENERATED = "generated"              # Created by builder/integrator during build
    USER_OWNED = "user_owned"            # Reserved for user post-build, builder must not touch


# ---------------------------------------------------------------------------
# Registry entry — one per template
# ---------------------------------------------------------------------------

class RegistryEntry(BaseModel):
    """A single template in the registry.

    Commit-pinned: pinned_commit is a content hash of the template
    directory, not a branch name. Resolution is deterministic.
    """
    model_config = {"extra": "forbid"}

    template_id: str = Field(description="Unique ID, e.g. 'python_cli_v1'")
    archetype: Archetype
    source_type: Literal["local"] = "local"
    source_path: str = Field(description="Path to template directory (relative to package)")
    pinned_commit: str = Field(description="Content hash of template dir (SHA-256)")
    template_version: str = Field(default="1")
    manifest_path: str = Field(
        default="ownership.json",
        description="Relative path to ownership manifest within the template"
    )


# ---------------------------------------------------------------------------
# Ownership manifest — written into output project
# ---------------------------------------------------------------------------

class OwnershipManifest(BaseModel):
    """Declares who owns each file path in the output project.

    Written during template materialization. Consulted before every
    file write during build. The source of truth for write permissions.
    """
    model_config = {"extra": "forbid"}

    schema_version: Literal["1"] = SCHEMA_VERSION

    # Provenance
    template_id: str
    pinned_commit: str
    contract_hash: str
    materialized_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Path → owner mapping
    # Every template file must be listed. Unlisted paths are new files
    # created by the builder during build — those are GENERATED.
    files: dict[str, FileOwner] = Field(
        default_factory=dict,
        description="relative_path -> FileOwner. Only builder-created files may be unlisted."
    )

    def owner_of(self, path: str) -> FileOwner:
        """Get the owner of a path. Unlisted paths (builder-created) are GENERATED."""
        return self.files.get(path, FileOwner.GENERATED)

    def can_write(self, path: str) -> bool:
        """Check if the builder/integrator is allowed to write this path."""
        owner = self.owner_of(path)
        return owner in (FileOwner.BUILDER_OWNED, FileOwner.GENERATED)

    def is_locked(self, path: str) -> bool:
        """Check if a path is template-locked or user-owned (immutable)."""
        owner = self.owner_of(path)
        return owner in (FileOwner.TEMPLATE_LOCKED, FileOwner.USER_OWNED)
