"""Ownership enforcement: validates file writes against the manifest.

Fail-closed: if a write targets a locked or user-owned path, raise.
"""

from __future__ import annotations

from build_loop.templates.models import FileOwner, OwnershipManifest


class OwnershipViolationError(Exception):
    """Raised when a write violates the ownership manifest."""


def check_write_allowed(manifest: OwnershipManifest, relative_path: str) -> None:
    """Check if writing to relative_path is allowed by the manifest.

    Raises OwnershipViolationError if the path is template_locked or user_owned.
    Paths not in the manifest are implicitly GENERATED and allowed.
    """
    owner = manifest.owner_of(relative_path)

    if owner == FileOwner.TEMPLATE_LOCKED:
        raise OwnershipViolationError(
            f"Cannot write to template-locked path: {relative_path!r}. "
            f"This file is owned by the template and must not be modified."
        )

    if owner == FileOwner.USER_OWNED:
        raise OwnershipViolationError(
            f"Cannot write to user-owned path: {relative_path!r}. "
            f"This file is reserved for user modifications."
        )
