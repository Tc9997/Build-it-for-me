"""Shared runtime services abstraction.

Thin wrappers around existing capabilities that engines can depend on
without importing concrete implementations directly. This is a seam
for future extraction — current implementations are just delegation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from build_loop.safety import PathTraversalError, safe_output_path
from build_loop.schemas import BuildState


class ProjectIO:
    """Filesystem operations scoped to a project directory."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def safe_write(self, relative_path: str, content: str) -> Path:
        """Write a file within the project directory. Path-safe."""
        resolved = safe_output_path(self.output_dir, relative_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return resolved

    def read_project_files(self, max_file_size: int = 50000) -> dict[str, str]:
        """Read all project files into a dict."""
        skip = {".venv", "__pycache__", ".build_state", ".git", "node_modules"}
        files = {}
        for p in self.output_dir.rglob("*"):
            if p.is_file() and not any(s in str(p) for s in skip):
                try:
                    content = p.read_text(errors="replace")
                    if len(content) < max_file_size:
                        files[str(p.relative_to(self.output_dir))] = content
                except Exception:
                    pass
        return files

    def save_state(self, state: BuildState) -> None:
        """Persist build state to disk."""
        state_dir = self.output_dir / ".build_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "state.json").write_text(state.model_dump_json(indent=2))

    def load_state(self) -> BuildState | None:
        """Load persisted build state, or None if not found."""
        state_path = self.output_dir / ".build_state" / "state.json"
        if not state_path.exists():
            return None
        return BuildState.model_validate_json(state_path.read_text())


class RuntimeServices:
    """Bundle of shared services available to any engine.

    Currently a thin wrapper. Future: injectable, mockable, testable
    without touching the filesystem or making LLM calls.
    """

    def __init__(self, output_dir: str):
        self.project_io = ProjectIO(output_dir)
        self.output_dir = output_dir

    @property
    def venv_python(self) -> str | None:
        """Path to the project's venv python, or None if not installed."""
        venv = Path(self.output_dir) / ".venv" / "bin" / "python"
        return str(venv) if venv.exists() else None
