"""Safety utilities for the build loop.

All validation of model-generated output happens here. Every write path and
every command execution path must go through these checks.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path


class PathTraversalError(Exception):
    """Raised when a generated path escapes the project root."""


class UnsafeCommandError(Exception):
    """Raised when a generated command contains shell injection metacharacters."""


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

# Characters that have no business in a project file path
_BAD_PATH_CHARS = re.compile(r'[\x00]')


def safe_output_path(project_root: str | Path, relative_path: str) -> Path:
    """Resolve a model-generated relative path safely within the project root.

    Raises PathTraversalError if the path:
      - Is absolute
      - Contains .. traversal
      - Resolves outside the project root
      - Contains null bytes
    """
    project_root = Path(project_root).resolve()
    relative_path = relative_path.strip()

    if not relative_path:
        raise PathTraversalError("Empty file path")

    if _BAD_PATH_CHARS.search(relative_path):
        raise PathTraversalError(f"Path contains illegal characters: {relative_path!r}")

    # Reject absolute paths outright
    if relative_path.startswith("/") or relative_path.startswith("\\"):
        raise PathTraversalError(
            f"Absolute path not allowed: {relative_path!r}"
        )

    # Reject explicit .. components
    parts = Path(relative_path).parts
    if ".." in parts:
        raise PathTraversalError(
            f"Path traversal (..) not allowed: {relative_path!r}"
        )

    # Resolve and check containment
    resolved = (project_root / relative_path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise PathTraversalError(
            f"Path escapes project root: {relative_path!r} "
            f"resolves to {resolved} which is outside {project_root}"
        )

    return resolved


# ---------------------------------------------------------------------------
# Command safety
# ---------------------------------------------------------------------------

# Shell metacharacters that indicate injection attempts
# Only match metacharacters that enable injection *outside* of quotes.
# Parentheses and brackets inside quoted arguments are harmless after shlex.split.
_SHELL_META = re.compile(r'[;|&`$<>\\]|&&|\|\|')


def safe_command(command: str) -> list[str]:
    """Parse a command string into an argv list, rejecting shell metacharacters.

    Returns the parsed argv suitable for subprocess.run(..., shell=False).
    Raises UnsafeCommandError if the command contains shell injection patterns.
    """
    command = command.strip()
    if not command:
        raise UnsafeCommandError("Empty command")

    # Check for shell metacharacters before parsing
    if _SHELL_META.search(command):
        raise UnsafeCommandError(
            f"Command contains shell metacharacters: {command!r}"
        )

    try:
        argv = shlex.split(command)
    except ValueError as e:
        raise UnsafeCommandError(f"Malformed command string: {command!r} ({e})")

    if not argv:
        raise UnsafeCommandError("Command parsed to empty argv")

    return argv
