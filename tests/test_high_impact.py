"""Tests for signal derivation, archetype checks, and README validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from build_loop.analysis.archetype_checks import run_archetype_checks
from build_loop.analysis.readme_validation import validate_readme
from build_loop.analysis.signal_derivation import derive_signals, merge_signals
from build_loop.contract import BuildContract, CliExitSignal, FileExistsSignal, ImportCheckSignal


def _make_contract(**overrides) -> BuildContract:
    defaults = dict(
        project_name="test_pkg", summary="test",
        goals=["test"], acceptance_criteria=["test"],
        archetype="python_cli",
    )
    defaults.update(overrides)
    return BuildContract(**defaults)


# =========================================================================
# Signal derivation
# =========================================================================

class TestSignalDerivation:

    def test_universal_signals(self):
        signals = derive_signals(_make_contract())
        types = [type(s).__name__ for s in signals]
        assert "ImportCheckSignal" in types
        assert "FileExistsSignal" in types

    def test_python_cli_has_init_signal(self):
        """python_cli derives __init__.py existence check, not python -m --help."""
        signals = derive_signals(_make_contract(archetype="python_cli"))
        file_signals = [s for s in signals if isinstance(s, FileExistsSignal)]
        assert any("__init__.py" in s.file_path for s in file_signals)
        # Should NOT have python -m --help (that's archetype_checks' job)
        cli_exits = [s for s in signals if isinstance(s, CliExitSignal)]
        assert not any("--help" in (s.args or []) for s in cli_exits)

    def test_python_cli_no_runtime_cli_checks_in_derivation(self):
        """Runtime CLI checks (--help, version) belong in archetype_checks, not derivation."""
        signals = derive_signals(_make_contract(
            archetype="python_cli",
            goals=["CLI with version command"],
        ))
        cli_exits = [s for s in signals if isinstance(s, CliExitSignal)]
        assert len(cli_exits) == 0  # No CLI runtime checks in derivation

    def test_fastapi_service_signals(self):
        signals = derive_signals(_make_contract(archetype="fastapi_service"))
        types = [type(s).__name__ for s in signals]
        assert "ImportCheckSignal" in types
        assert any(isinstance(s, FileExistsSignal) and "app.py" in s.file_path for s in signals)

    def test_typed_goal_adds_py_typed_signal(self):
        signals = derive_signals(_make_contract(
            goals=["typed library with py.typed marker"],
        ))
        file_signals = [s for s in signals if isinstance(s, FileExistsSignal)]
        assert any("py.typed" in s.file_path for s in file_signals)

    def test_all_signals_are_well_structured(self):
        """No signal has spaces in command field."""
        signals = derive_signals(_make_contract(
            archetype="python_cli",
            goals=["CLI with version, schema, validate, registry commands"],
        ))
        for s in signals:
            if isinstance(s, CliExitSignal):
                assert " " not in s.command, f"Signal has spaces in command: {s.command}"

    def test_unsupported_archetype_only_universal(self):
        signals = derive_signals(_make_contract(archetype="unsupported"))
        # Only universal signals (import + pyproject.toml)
        assert len(signals) == 2

    def test_merge_deduplicates(self):
        """Merge doesn't duplicate signals with same type+description."""
        derived = [ImportCheckSignal(description="pkg importable", module_name="pkg")]
        llm = [ImportCheckSignal(description="pkg importable", module_name="pkg")]
        merged = merge_signals(derived, llm)
        assert len(merged) == 1

    def test_merge_keeps_project_specific(self):
        """LLM-generated project-specific signals survive merge."""
        derived = [ImportCheckSignal(description="pkg importable", module_name="pkg")]
        llm = [FileExistsSignal(description="output.csv exists", file_path="output.csv")]
        merged = merge_signals(derived, llm)
        assert len(merged) == 2
        assert any("output.csv" in getattr(s, "file_path", "") for s in merged)

    def test_merge_drops_malformed_llm_signals(self):
        """LLM signals with spaces in command are dropped during merge."""
        derived = [ImportCheckSignal(description="pkg importable", module_name="pkg")]
        # This would fail contract validation but could exist in older state
        from unittest.mock import MagicMock
        bad = MagicMock()
        bad.type = "cli_exit"
        bad.description = "bad signal"
        bad.command = "python -m foo"
        bad.args = []
        merged = merge_signals(derived, [bad])
        assert len(merged) == 1  # Bad signal dropped


# =========================================================================
# Archetype checks
# =========================================================================

class TestArchetypeChecks:

    def test_python_cli_valid_package(self, tmp_path):
        pkg = tmp_path / "my_tool"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("__version__ = '1.0'\nfrom .core import main\n")
        (pkg / "core.py").write_text("def main():\n    print('hello')\n")

        results = run_archetype_checks(str(tmp_path), "python_cli")
        assert any(r.passed and "Package directory" in r.description for r in results)
        assert any(r.passed and "real content" in r.description for r in results)

    def test_python_cli_no_package(self, tmp_path):
        results = run_archetype_checks(str(tmp_path), "python_cli")
        assert any(not r.passed and "Package directory" in r.description for r in results)

    def test_python_cli_stub_init(self, tmp_path):
        pkg = tmp_path / "my_tool"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("# TODO\n")

        results = run_archetype_checks(str(tmp_path), "python_cli")
        assert any(not r.passed and "stub" in r.description.lower() or "real content" in r.description
                   for r in results if not r.passed)

    def test_fastapi_valid_package(self, tmp_path):
        pkg = tmp_path / "my_api"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("from .app import app\n")
        (pkg / "app.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")

        results = run_archetype_checks(str(tmp_path), "fastapi_service")
        assert any(r.passed and "FastAPI" in r.description for r in results)

    def test_unsupported_returns_empty(self, tmp_path):
        results = run_archetype_checks(str(tmp_path), "unsupported")
        assert results == []


# =========================================================================
# README validation
# =========================================================================

class TestReadmeValidation:

    def test_valid_readme(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "# My Tool\n\n```python\nfrom my_pkg import MyModel\n```\n"
        )
        exports = {
            "mod_a": {"exported_classes": ["MyModel"], "exported_functions": []},
        }
        errors = validate_readme(tmp_path, exports)
        assert errors == []

    def test_readme_references_nonexistent_class(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "# My Tool\n\n```python\nfrom my_pkg import FakeClass\n```\n"
        )
        exports = {
            "mod_a": {"exported_classes": ["RealClass"], "exported_functions": []},
        }
        errors = validate_readme(tmp_path, exports)
        assert any("FakeClass" in e for e in errors)

    def test_no_readme_is_ok(self, tmp_path):
        errors = validate_readme(tmp_path, {"mod_a": {"exported_classes": ["X"]}})
        assert errors == []

    def test_no_code_blocks_is_ok(self, tmp_path):
        (tmp_path / "README.md").write_text("# My Tool\n\nJust text, no code.\n")
        errors = validate_readme(tmp_path, {"mod_a": {"exported_classes": ["X"]}})
        assert errors == []

    def test_builtin_imports_not_flagged(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "```python\nfrom pydantic import BaseModel, Field\n```\n"
        )
        exports = {"mod_a": {"exported_classes": [], "exported_functions": []}}
        errors = validate_readme(tmp_path, exports)
        assert errors == []
