"""Tests for post-write deterministic checks."""

from pathlib import Path

import pytest

from build_loop.analysis.post_write import run_post_write_checks


class TestPostWriteChecks:

    def test_valid_cli_project(self, tmp_path):
        """A valid python_cli project passes all checks."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "0.1.0"\n'
            '[project.scripts]\ntest = "src.cli:main"\n'
        )
        src = tmp_path / "src"
        src.mkdir()
        (src / "__init__.py").write_text("")
        (src / "cli.py").write_text("def main():\n    pass\n")

        result = run_post_write_checks(str(tmp_path), "python_cli")
        assert result.passed
        assert any("Entry point" in c for c in result.checks)
        assert any("CLI entry point found" in c for c in result.checks)

    def test_missing_entry_point_module(self, tmp_path):
        """Entry point referencing missing module fails."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "0.1.0"\n'
            '[project.scripts]\ntest = "nonexistent.cli:main"\n'
        )

        result = run_post_write_checks(str(tmp_path), "python_cli")
        assert not result.passed
        assert any("not found" in e for e in result.errors)

    def test_missing_entry_point_function(self, tmp_path):
        """Entry point referencing missing function fails."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "0.1.0"\n'
            '[project.scripts]\ntest = "src.cli:nonexistent"\n'
        )
        src = tmp_path / "src"
        src.mkdir()
        (src / "cli.py").write_text("def other():\n    pass\n")

        result = run_post_write_checks(str(tmp_path), "python_cli")
        assert not result.passed
        assert any("nonexistent" in e for e in result.errors)

    def test_invalid_toml(self, tmp_path):
        """Malformed pyproject.toml fails."""
        (tmp_path / "pyproject.toml").write_text("[broken toml\n")

        result = run_post_write_checks(str(tmp_path))
        assert not result.passed
        assert any("parse error" in e for e in result.errors)

    def test_fastapi_archetype(self, tmp_path):
        """FastAPI project needs app.py with FastAPI."""
        pkg = tmp_path / "myapp"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "app.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")

        result = run_post_write_checks(str(tmp_path), "fastapi_service")
        assert result.passed
        assert any("FastAPI app found" in c for c in result.checks)
