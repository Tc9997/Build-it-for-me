"""Deterministic export analysis for built module artifacts.

Uses Python AST to extract classes, functions, constants, and import
statements from generated code. No LLM. Runs on file content strings
before they are written to disk.
"""

from __future__ import annotations

import ast
from typing import Any

from pydantic import BaseModel, Field

from build_loop.schemas import BuildArtifact


class ModuleExports(BaseModel):
    """Structured export metadata for a single built module."""
    module_id: str
    files: list[str] = Field(default_factory=list)
    exported_classes: list[str] = Field(default_factory=list)
    exported_functions: list[str] = Field(default_factory=list)
    exported_constants: list[str] = Field(default_factory=list)
    import_statements: list[str] = Field(default_factory=list)
    syntax_valid: bool = True
    parse_errors: list[str] = Field(default_factory=list)


def analyze_artifact(artifact: BuildArtifact) -> ModuleExports:
    """Extract export metadata from a BuildArtifact's file contents.

    Analyzes all .py files in the artifact using AST parsing.
    Non-Python files are recorded in the file list but not parsed.
    """
    exports = ModuleExports(module_id=artifact.module_id)

    all_files = {**artifact.files, **artifact.tests}
    exports.files = sorted(all_files.keys())

    for path, content in all_files.items():
        if not path.endswith(".py"):
            continue
        _analyze_python_file(path, content, exports)

    return exports


def _analyze_python_file(path: str, content: str, exports: ModuleExports) -> None:
    """Parse a single Python file and extract symbols."""
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError as e:
        exports.syntax_valid = False
        exports.parse_errors.append(f"{path}: {e}")
        return

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            exports.exported_classes.append(node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            exports.exported_functions.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    exports.exported_constants.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper():
                exports.exported_constants.append(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                exports.import_statements.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(a.name for a in node.names)
            exports.import_statements.append(f"from {module} import {names}")
