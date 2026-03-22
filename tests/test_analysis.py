"""Tests for build analysis: export extraction, framework hints,
pre-integration validation, and syntax screening."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from build_loop.analysis.exports import ModuleExports, analyze_artifact
from build_loop.analysis.contract_validation import validate_pre_integration
from build_loop.analysis.framework_hints import get_framework_hints
from build_loop.common.pipeline import _screen_syntax, _gather_dependency_exports
from build_loop.schemas import (
    BuildArtifact,
    BuildPlan,
    BuildState,
    ModuleSpec,
    TaskSize,
)


# =========================================================================
# Export analysis
# =========================================================================

class TestExportAnalyzer:
    """AST-based export extraction from built module code."""

    def test_extracts_classes_from_production(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "class MyModel:\n    pass\n\nclass OtherModel:\n    pass\n"},
        )
        exports = analyze_artifact(artifact)
        assert "MyModel" in exports.exported_classes
        assert "OtherModel" in exports.exported_classes
        assert exports.syntax_valid

    def test_extracts_functions_from_production(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "def process():\n    pass\n\nasync def fetch():\n    pass\n"},
        )
        exports = analyze_artifact(artifact)
        assert "process" in exports.exported_functions
        assert "fetch" in exports.exported_functions

    def test_extracts_constants(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "VERSION = '1.0'\nMAX_RETRIES = 3\n"},
        )
        exports = analyze_artifact(artifact)
        assert "VERSION" in exports.exported_constants
        assert "MAX_RETRIES" in exports.exported_constants

    def test_extracts_imports(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "from pydantic import BaseModel, Field\nimport json\n"},
        )
        exports = analyze_artifact(artifact)
        assert "from pydantic import BaseModel, Field" in exports.import_statements
        assert "import json" in exports.import_statements

    def test_detects_syntax_errors(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "def broken(\n"},
        )
        exports = analyze_artifact(artifact)
        assert not exports.syntax_valid
        assert len(exports.parse_errors) > 0

    def test_records_production_and_test_files_separately(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "pass", "README.md": "# Hi"},
            tests={"tests/test_a.py": "pass"},
        )
        exports = analyze_artifact(artifact)
        assert "mod_a.py" in exports.files
        assert "README.md" in exports.files
        assert "tests/test_a.py" in exports.test_files
        # Test files are NOT in the production files list
        assert "tests/test_a.py" not in exports.files

    def test_skips_non_python_files(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"data.json": '{"key": "value"}'},
        )
        exports = analyze_artifact(artifact)
        assert exports.syntax_valid
        assert exports.exported_classes == []

    def test_test_symbols_not_in_production_exports(self):
        """Test file classes/functions must NOT appear in exported_classes/functions."""
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "class Foo: pass"},
            tests={"tests/test_a.py": "class TestFoo: pass\ndef test_foo(): pass"},
        )
        exports = analyze_artifact(artifact)
        # Production export
        assert "Foo" in exports.exported_classes
        # Test symbols are in test_classes/test_functions, NOT production
        assert "TestFoo" in exports.test_classes
        assert "test_foo" in exports.test_functions
        assert "TestFoo" not in exports.exported_classes
        assert "test_foo" not in exports.exported_functions

    def test_dependency_resolution_uses_production_only(self):
        """A module that only exports symbols from tests has empty production exports."""
        artifact = BuildArtifact(
            module_id="mod_a",
            files={},  # No production files
            tests={"tests/test_a.py": "class FakeExport: pass"},
        )
        exports = analyze_artifact(artifact)
        assert exports.exported_classes == []
        assert "FakeExport" in exports.test_classes


# =========================================================================
# Framework hints
# =========================================================================

class TestFrameworkHints:
    """Static framework guidance blocks."""

    def test_pydantic_hints_present(self):
        hints = get_framework_hints(["pydantic>=2.0"])
        assert "field_validator" in hints
        assert "model_dump" in hints
        assert "@validator" in hints

    def test_fastapi_hints_present(self):
        hints = get_framework_hints(["fastapi>=0.110"])
        assert "APIRouter" in hints

    def test_no_hints_for_unknown_stack(self):
        hints = get_framework_hints(["some-unknown-lib"])
        assert hints == ""

    def test_multiple_frameworks(self):
        hints = get_framework_hints(["pydantic>=2.0", "fastapi>=0.110"])
        assert "field_validator" in hints
        assert "APIRouter" in hints

    def test_case_insensitive(self):
        hints = get_framework_hints(["Pydantic>=2.0"])
        assert "field_validator" in hints


# =========================================================================
# Syntax screening
# =========================================================================

class TestSyntaxScreening:
    """Syntax and import screening before LLM review."""

    def test_clean_code_passes(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "class Foo:\n    pass\n"},
        )
        issues = _screen_syntax(artifact)
        assert issues == []

    def test_broken_code_caught(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "def broken(\n"},
        )
        issues = _screen_syntax(artifact)
        assert len(issues) > 0
        assert "syntax" in issues[0].lower() or "Syntax" in issues[0]

    def test_unresolved_internal_import_caught(self):
        """Import of a project-internal module that doesn't exist is caught."""
        artifact = BuildArtifact(
            module_id="mod_b",
            files={
                "src/__init__.py": "",
                "src/mod_b.py": "from src.provider import Foo\n",
                # src/provider.py does NOT exist
            },
        )
        issues = _screen_syntax(artifact)
        assert any("unresolved" in i.lower() for i in issues)
        assert any("src.provider" in i for i in issues)

    def test_valid_internal_import_passes(self):
        """Import of an existing project-internal module is not flagged."""
        artifact = BuildArtifact(
            module_id="mod_b",
            files={
                "src/__init__.py": "",
                "src/provider.py": "class Foo: pass\n",
                "src/consumer.py": "from src.provider import Foo\n",
            },
        )
        issues = _screen_syntax(artifact)
        assert issues == []

    def test_stdlib_import_not_flagged(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "import json\nfrom pathlib import Path\n"},
        )
        issues = _screen_syntax(artifact)
        assert issues == []

    def test_third_party_import_not_flagged(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "from pydantic import BaseModel\nimport requests\n"},
        )
        issues = _screen_syntax(artifact)
        assert issues == []


class TestUnresolvedImportDetection:
    """Detailed tests for _find_unresolved_imports."""

    def test_detects_missing_sibling_module(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={
                "src/__init__.py": "",
                "src/main.py": "from src.missing_module import Thing\n",
            },
        )
        exports = analyze_artifact(artifact)
        assert "from src.missing_module import Thing" in exports.unresolved_imports

    def test_existing_sibling_not_flagged(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={
                "src/__init__.py": "",
                "src/utils.py": "def helper(): pass\n",
                "src/main.py": "from src.utils import helper\n",
            },
        )
        exports = analyze_artifact(artifact)
        assert exports.unresolved_imports == []

    def test_bare_third_party_not_flagged(self):
        """A bare import like 'import requests' is not flagged as unresolved."""
        artifact = BuildArtifact(
            module_id="mod_a",
            files={"mod_a.py": "import requests\nfrom fastapi import FastAPI\n"},
        )
        exports = analyze_artifact(artifact)
        assert exports.unresolved_imports == []

    def test_deep_nested_missing_module(self):
        artifact = BuildArtifact(
            module_id="mod_a",
            files={
                "pkg/__init__.py": "",
                "pkg/sub/__init__.py": "",
                "pkg/sub/real.py": "X = 1\n",
                "pkg/main.py": "from pkg.sub.fake import Y\n",
            },
        )
        exports = analyze_artifact(artifact)
        assert any("pkg.sub.fake" in u for u in exports.unresolved_imports)

    def test_relative_imports_not_flagged(self):
        """Relative imports (from . import foo) should never be flagged as unresolved."""
        artifact = BuildArtifact(
            module_id="mod_a",
            files={
                "pkg/__init__.py": "",
                "pkg/main.py": "from . import utils\nfrom .. import parent\n",
                "pkg/utils.py": "X = 1\n",
            },
        )
        exports = analyze_artifact(artifact)
        assert exports.unresolved_imports == []


# =========================================================================
# Dependency context gathering
# =========================================================================

class TestDependencyContext:
    """Downstream modules receive actual built dependency context."""

    def test_gathers_from_earlier_batches(self):
        module = ModuleSpec(
            id="mod_b", name="B", description="test", size=TaskSize.SMALL,
            dependencies=["mod_a"],
        )
        module_specs = {
            "mod_a": ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL),
            "mod_b": module,
        }
        all_exports = {
            "mod_a": ModuleExports(
                module_id="mod_a",
                files=["mod_a.py"],
                exported_classes=["BaseModel"],
            ),
        }
        result = _gather_dependency_exports(module, module_specs, all_exports)
        assert "mod_a" in result
        assert "BaseModel" in result["mod_a"].exported_classes

    def test_ignores_unbuilt_dependencies(self):
        module = ModuleSpec(
            id="mod_b", name="B", description="test", size=TaskSize.SMALL,
            dependencies=["mod_a", "mod_c"],
        )
        module_specs = {"mod_b": module}
        all_exports = {
            "mod_a": ModuleExports(module_id="mod_a", files=["a.py"]),
        }
        result = _gather_dependency_exports(module, module_specs, all_exports)
        assert "mod_a" in result
        assert "mod_c" not in result

    def test_no_dependencies_returns_empty(self):
        module = ModuleSpec(
            id="mod_a", name="A", description="test", size=TaskSize.SMALL,
            dependencies=[],
        )
        result = _gather_dependency_exports(module, {}, {})
        assert result == {}


# =========================================================================
# Pre-integration contract validation
# =========================================================================

class TestPreIntegrationValidation:
    """Validates built modules against planned contracts before integration."""

    def test_valid_modules_pass(self):
        plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            modules=[
                ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL,
                           file_paths=["mod_a.py"]),
            ],
            build_order=[["mod_a"]],
        )
        exports = {
            "mod_a": ModuleExports(
                module_id="mod_a", files=["mod_a.py"],
                exported_classes=["Foo"], syntax_valid=True,
            ),
        }
        result = validate_pre_integration(plan, exports)
        assert result.valid

    def test_syntax_errors_block(self):
        plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            modules=[ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL)],
            build_order=[["mod_a"]],
        )
        exports = {
            "mod_a": ModuleExports(
                module_id="mod_a", syntax_valid=False,
                parse_errors=["mod_a.py: invalid syntax"],
            ),
        }
        result = validate_pre_integration(plan, exports)
        assert not result.valid
        assert any("syntax" in e.lower() for e in result.errors)

    def test_missing_planned_file_blocks(self):
        """Missing planned production file is a blocking error, not a warning."""
        plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            modules=[
                ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL,
                           file_paths=["mod_a.py", "mod_a_utils.py"]),
            ],
            build_order=[["mod_a"]],
        )
        exports = {
            "mod_a": ModuleExports(
                module_id="mod_a", files=["mod_a.py"],
                syntax_valid=True,
            ),
        }
        result = validate_pre_integration(plan, exports)
        assert not result.valid
        assert any("mod_a_utils.py" in e for e in result.errors)

    def test_same_batch_dependency_blocks(self):
        """Two modules in the same batch with a dependency edge is a blocking error."""
        plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            modules=[
                ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL),
                ModuleSpec(id="mod_b", name="B", description="test", size=TaskSize.SMALL,
                           dependencies=["mod_a"]),
            ],
            build_order=[["mod_a", "mod_b"]],  # Same batch!
        )
        exports = {
            "mod_a": ModuleExports(module_id="mod_a", syntax_valid=True),
            "mod_b": ModuleExports(module_id="mod_b", syntax_valid=True),
        }
        result = validate_pre_integration(plan, exports)
        assert not result.valid
        assert any("same" in e.lower() and "batch" in e.lower() for e in result.errors)

    def test_cross_batch_dependency_ok(self):
        """Dependencies across batches are fine."""
        plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            modules=[
                ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL),
                ModuleSpec(id="mod_b", name="B", description="test", size=TaskSize.SMALL,
                           dependencies=["mod_a"]),
            ],
            build_order=[["mod_a"], ["mod_b"]],  # Different batches
        )
        exports = {
            "mod_a": ModuleExports(module_id="mod_a", syntax_valid=True,
                                   exported_classes=["Foo"]),
            "mod_b": ModuleExports(module_id="mod_b", syntax_valid=True),
        }
        result = validate_pre_integration(plan, exports)
        assert result.valid

    def test_unbuilt_dependency_warns(self):
        plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            modules=[
                ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL),
                ModuleSpec(id="mod_b", name="B", description="test", size=TaskSize.SMALL,
                           dependencies=["mod_a"]),
            ],
            build_order=[["mod_a"], ["mod_b"]],
        )
        exports = {
            "mod_b": ModuleExports(module_id="mod_b", syntax_valid=True),
        }
        result = validate_pre_integration(plan, exports)
        assert any("mod_a" in w and "not built" in w for w in result.warnings)

    def test_test_only_exports_not_counted(self):
        """A dependency that only exports from test files has empty production exports."""
        plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            modules=[
                ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL),
                ModuleSpec(id="mod_b", name="B", description="test", size=TaskSize.SMALL,
                           dependencies=["mod_a"]),
            ],
            build_order=[["mod_a"], ["mod_b"]],
        )
        exports = {
            "mod_a": ModuleExports(
                module_id="mod_a", syntax_valid=True,
                exported_classes=[],  # No production exports
                test_classes=["FakeExport"],  # Only in tests
            ),
            "mod_b": ModuleExports(module_id="mod_b", syntax_valid=True),
        }
        result = validate_pre_integration(plan, exports)
        # Should warn that mod_a has no production exports
        assert any("no production exports" in w for w in result.warnings)


# =========================================================================
# Builder receives dependency exports
# =========================================================================

class TestBuilderDependencyContext:
    """Builder prompt includes actual dependency exports."""

    def test_builder_accepts_dependency_exports(self):
        from build_loop.agents.builder import BuilderAgent
        agent = BuilderAgent()
        agent.call_json = MagicMock(return_value={
            "module_id": "mod_b",
            "files": {"mod_b.py": "from mod_a import Foo"},
            "tests": {},
            "notes": "",
        })

        module = ModuleSpec(
            id="mod_b", name="B", description="test", size=TaskSize.SMALL,
            dependencies=["mod_a"],
        )
        plan = BuildPlan(
            project_name="test", description="test",
            tech_stack=["pydantic>=2.0"],
        )
        dep_exports = {
            "mod_a": ModuleExports(
                module_id="mod_a", files=["mod_a.py"],
                exported_classes=["Foo", "Bar"],
            ),
        }

        artifact = agent.run(module, plan, dependency_exports=dep_exports)
        assert artifact.module_id == "mod_b"

        prompt = agent.call_json.call_args[0][0]
        assert "DEPENDENCY CONTEXT" in prompt
        assert "Foo" in prompt
        assert "Bar" in prompt

    def test_builder_includes_framework_hints(self):
        from build_loop.agents.builder import BuilderAgent
        agent = BuilderAgent()
        agent.call_json = MagicMock(return_value={
            "module_id": "mod_a",
            "files": {"mod_a.py": "pass"},
            "tests": {},
            "notes": "",
        })

        module = ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL)
        plan = BuildPlan(
            project_name="test", description="test",
            tech_stack=["pydantic>=2.0", "fastapi"],
        )

        agent.run(module, plan)
        prompt = agent.call_json.call_args[0][0]
        assert "field_validator" in prompt
        assert "model_dump" in prompt


# =========================================================================
# Dependency installation helper
# =========================================================================

class TestDependencyInstall:
    """Centralized dep install strips version specs safely."""

    def test_strip_simple_version(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("pytest>=7.0") == "pytest"

    def test_strip_double_equals(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("pydantic==2.5.0") == "pydantic"

    def test_strip_tilde(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("requests~=2.31") == "requests"

    def test_strip_less_than(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("numpy<2.0") == "numpy"

    def test_strip_not_equals(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("setuptools!=60.0") == "setuptools"

    def test_strip_extras(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("uvicorn[standard]>=0.29") == "uvicorn"

    def test_strip_extras_only(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("pydantic[email]") == "pydantic"

    def test_no_version_passthrough(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("requests") == "requests"

    def test_empty_string(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("") == ""

    def test_complex_spec(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec("typing-extensions>=4.0,<5.0") == "typing-extensions"

    def test_semicolon_env_marker(self):
        from build_loop.common.pipeline import _strip_version_spec
        assert _strip_version_spec('importlib-metadata>=1.0;python_version<"3.8"') == "importlib-metadata"

    def test_install_dependencies_calls_executor(self):
        from build_loop.common.pipeline import install_dependencies
        executor = MagicMock()
        executor.run_command = MagicMock(return_value=MagicMock(success=True))
        state = BuildState()

        install_dependencies(
            ["pytest>=7.0", "pydantic[email]>=2.0", "requests"],
            executor,
            lambda cmd: cmd,  # identity venv_cmd
            state,
        )

        # All three deps installed, version specs stripped
        calls = executor.run_command.call_args_list
        assert len(calls) == 3
        assert calls[0][0][0] == "pip install pytest"
        assert calls[1][0][0] == "pip install pydantic"
        assert calls[2][0][0] == "pip install requests"
