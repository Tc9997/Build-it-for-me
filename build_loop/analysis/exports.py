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


def _find_unresolved_imports(exports: ModuleExports, artifact: BuildArtifact) -> list[str]:
    """Find imports that reference modules not in stdlib or the project itself.

    Returns a list of module references that might be unresolved.
    This is a heuristic — it can't know about installed third-party packages,
    but it can catch imports from sibling modules that don't exist in the artifact.
    """
    unresolved = []
    # Collect all file-based module paths in the artifact
    project_modules = set()
    for path in list(artifact.files.keys()) + list(artifact.tests.keys()):
        if path.endswith(".py"):
            # "src/foo/bar.py" → "src.foo.bar" and "src.foo" and "src"
            parts = path.replace("/", ".").removesuffix(".py").split(".")
            for i in range(len(parts)):
                project_modules.add(".".join(parts[:i + 1]))

    for imp in exports.import_statements:
        # Extract the root module name
        if imp.startswith("from "):
            # "from foo.bar import Baz" → "foo"
            module_path = imp.split(" ")[1]
            root = module_path.split(".")[0]
        elif imp.startswith("import "):
            root = imp.split(" ")[1].split(".")[0]
        else:
            continue

        if root and root not in _STDLIB_MODULES and root not in project_modules:
            # Could be a third-party package (not necessarily unresolved)
            # Only flag as unresolved if it looks like an internal project import
            # that doesn't match any produced file
            if "." in (imp.split(" ")[1] if imp.startswith("import ") else imp.split(" ")[1]):
                full_module = imp.split(" ")[1].split(".")[0]
                # Check if any artifact file matches this root
                if full_module not in project_modules and full_module not in _STDLIB_MODULES:
                    pass  # Third-party — can't validate without pip freeze

    return unresolved
