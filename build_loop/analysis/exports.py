"""Deterministic export analysis for built module artifacts.

Uses Python AST to extract classes, functions, constants, and import
statements from generated code. No LLM. Runs on file content strings
before they are written to disk.

Production files (artifact.files) and test files (artifact.tests) are
analyzed separately. Only production exports count for dependency
resolution — test symbols are never treated as public API.
"""

from __future__ import annotations

import ast
from typing import Any

from pydantic import BaseModel, Field

from build_loop.schemas import BuildArtifact


class FileExports(BaseModel):
    """Exports from a single file."""
    classes: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    constants: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)


class ModuleExports(BaseModel):
    """Structured export metadata for a single built module.

    exported_classes/functions/constants are from PRODUCTION files only.
    test_classes/functions are from test files (not public API).
    """
    module_id: str
    files: list[str] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)

    # Production exports (public API — used for dependency resolution)
    exported_classes: list[str] = Field(default_factory=list)
    exported_functions: list[str] = Field(default_factory=list)
    exported_constants: list[str] = Field(default_factory=list)
    import_statements: list[str] = Field(default_factory=list)

    # Test-only symbols (NOT public API)
    test_classes: list[str] = Field(default_factory=list)
    test_functions: list[str] = Field(default_factory=list)

    # Unresolved imports (imports referencing modules not in stdlib/known)
    unresolved_imports: list[str] = Field(default_factory=list)

    syntax_valid: bool = True
    parse_errors: list[str] = Field(default_factory=list)


def analyze_artifact(artifact: BuildArtifact) -> ModuleExports:
    """Extract export metadata from a BuildArtifact's file contents.

    Production files (artifact.files) contribute to public exports.
    Test files (artifact.tests) are analyzed separately and their
    symbols are stored in test_classes/test_functions only.
    """
    exports = ModuleExports(module_id=artifact.module_id)
    exports.files = sorted(artifact.files.keys())
    exports.test_files = sorted(artifact.tests.keys())

    # Analyze production files → public exports
    for path, content in artifact.files.items():
        if not path.endswith(".py"):
            continue
        file_exports = _analyze_python_file(path, content, exports)
        if file_exports:
            exports.exported_classes.extend(file_exports.classes)
            exports.exported_functions.extend(file_exports.functions)
            exports.exported_constants.extend(file_exports.constants)
            exports.import_statements.extend(file_exports.imports)

    # Analyze test files → test-only symbols (not public API)
    for path, content in artifact.tests.items():
        if not path.endswith(".py"):
            continue
        file_exports = _analyze_python_file(path, content, exports)
        if file_exports:
            exports.test_classes.extend(file_exports.classes)
            exports.test_functions.extend(file_exports.functions)
            # Test imports still go into import_statements for screening
            exports.import_statements.extend(file_exports.imports)

    # Detect unresolved imports
    exports.unresolved_imports = _find_unresolved_imports(exports, artifact)

    return exports


def _analyze_python_file(path: str, content: str, exports: ModuleExports) -> FileExports | None:
    """Parse a single Python file and extract symbols. Returns None on syntax error."""
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError as e:
        exports.syntax_valid = False
        exports.parse_errors.append(f"{path}: {e}")
        return None

    result = FileExports()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            result.classes.append(node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            result.functions.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    result.constants.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper():
                result.constants.append(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result.imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(a.name for a in node.names)
            result.imports.append(f"from {module} import {names}")

    return result


# Standard library module names (Python 3.11+). Not exhaustive but covers
# the most common ones to avoid false positives in unresolved import detection.
_STDLIB_MODULES = frozenset({
    "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "enum", "functools", "hashlib",
    "http", "importlib", "inspect", "io", "itertools", "json", "logging",
    "math", "os", "pathlib", "pickle", "platform", "pprint", "random", "re",
    "shlex", "shutil", "signal", "socket", "sqlite3", "string", "subprocess",
    "sys", "tempfile", "textwrap", "threading", "time", "traceback", "typing",
    "typing_extensions", "unittest", "urllib", "uuid", "warnings", "xml",
    "__future__", "types", "operator", "secrets", "struct", "zlib", "gzip",
})


def _find_unresolved_imports(
    exports: ModuleExports,
    artifact: BuildArtifact,
    known_third_party: frozenset[str] | None = None,
) -> list[str]:
    """Find imports of project-internal modules that don't exist in the artifact.

    Catches: `from src.provider import Foo` when `src/provider.py` doesn't exist.

    Does NOT flag:
    - Standard library imports
    - Known third-party packages (pydantic, fastapi, pytest, etc.)

    Returns a list of unresolved import strings.
    """
    if known_third_party is None:
        known_third_party = _KNOWN_THIRD_PARTY

    unresolved = []

    # Build the set of module paths that exist in this artifact
    project_modules = set()
    for path in list(artifact.files.keys()) + list(artifact.tests.keys()):
        if path.endswith(".py"):
            # "src/foo/bar.py" → "src.foo.bar", "src.foo", "src"
            parts = path.replace("/", ".").removesuffix(".py").split(".")
            for i in range(len(parts)):
                project_modules.add(".".join(parts[: i + 1]))
            # Also add __init__ package paths
            # "src/foo/__init__.py" → "src.foo"
            if parts[-1] == "__init__":
                project_modules.add(".".join(parts[:-1]))

    for imp in exports.import_statements:
        # Parse the import to get the full module path
        if imp.startswith("from "):
            # "from src.foo import Bar" → module_path = "src.foo"
            parts = imp.split()
            if len(parts) < 2:
                continue
            module_path = parts[1]
        elif imp.startswith("import "):
            parts = imp.split()
            if len(parts) < 2:
                continue
            module_path = parts[1].split(",")[0].strip()
        else:
            continue

        # Skip relative imports (from . import foo, from .. import bar)
        if module_path.startswith("."):
            continue

        root = module_path.split(".")[0]

        # Skip stdlib and known third-party
        if root in _STDLIB_MODULES or root in known_third_party:
            continue

        # Check if this looks like a project-internal import
        # A dotted import (src.foo.bar) whose root matches a project directory
        # is internal. A bare import (requests) with no matching project file
        # is likely third-party — skip it.
        if root in project_modules:
            # Internal import — check if the full path resolves
            if module_path not in project_modules:
                unresolved.append(imp)
        # else: bare import not matching any project file — likely third-party, skip

    return unresolved


# Known third-party packages. Not exhaustive but covers common ones
# to avoid false positives.
_KNOWN_THIRD_PARTY = frozenset({
    "pydantic", "fastapi", "uvicorn", "starlette", "httpx", "requests",
    "pytest", "click", "typer", "rich", "anthropic", "openai",
    "sqlalchemy", "alembic", "celery", "redis", "boto3", "flask",
    "django", "numpy", "pandas", "scipy", "sklearn", "torch",
    "dotenv", "yaml", "toml", "jinja2", "aiohttp", "websockets",
    "pydantic_settings", "cryptography", "jwt", "passlib", "bcrypt",
})
