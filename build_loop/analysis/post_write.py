"""Deterministic post-write checks run after files are written to disk.

Catches entry point wiring issues before the full verifier runs:
- pyproject.toml has valid TOML syntax
- console_scripts entry points reference importable modules
- archetype-specific checks (CLI has main(), service has app factory)
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PostWriteResult:
    """Result of post-write checks."""
    passed: bool = True
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_post_write_checks(
    output_dir: str,
    archetype: str = "",
) -> PostWriteResult:
    """Run deterministic checks after project files are written.

    No LLM. Just filesystem + subprocess checks.
    """
    result = PostWriteResult()
    out = Path(output_dir)

    # 1. pyproject.toml exists and parses
    pyproject = out / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            result.checks.append("pyproject.toml: valid TOML")

            # 2. Check console_scripts entry points
            scripts = data.get("project", {}).get("scripts", {})
            for name, target in scripts.items():
                _check_entry_point(out, name, target, result)

        except Exception as e:
            result.errors.append(f"pyproject.toml: parse error: {e}")
            result.passed = False
    else:
        result.checks.append("pyproject.toml: not present (ok for some projects)")

    # 3. Archetype-specific checks
    if archetype == "python_cli":
        _check_python_cli(out, result)
    elif archetype == "fastapi_service":
        _check_fastapi_service(out, result)

    # 4. Package importable check (after pip install -e . in setup)
    venv_python = out / ".venv" / "bin" / "python"
    if venv_python.exists():
        # Find the main package directory
        pkg_dirs = [d for d in out.iterdir()
                    if d.is_dir() and (d / "__init__.py").exists()
                    and d.name not in ("tests", "src", ".venv")]
        for pkg in pkg_dirs:
            _check_package_import(venv_python, pkg.name, result)

    if result.errors:
        result.passed = False

    return result


def _check_entry_point(out: Path, name: str, target: str, result: PostWriteResult) -> None:
    """Check that a console_scripts entry point is resolvable."""
    if ":" not in target:
        result.errors.append(f"Entry point '{name}': invalid format '{target}' (missing ':')")
        return

    module_path, func_name = target.rsplit(":", 1)

    # Convert module path to file path
    file_path = out / module_path.replace(".", "/")
    py_file = file_path.with_suffix(".py")
    pkg_init = file_path / "__init__.py"

    if py_file.exists():
        # Check the function exists in the file
        content = py_file.read_text()
        if f"def {func_name}" in content:
            result.checks.append(f"Entry point '{name}': {target} — found")
        else:
            result.errors.append(
                f"Entry point '{name}': function '{func_name}' not found in {py_file.relative_to(out)}"
            )
    elif pkg_init.exists():
        result.checks.append(f"Entry point '{name}': {target} — package exists")
    else:
        result.errors.append(
            f"Entry point '{name}': module '{module_path}' not found "
            f"(checked {py_file.relative_to(out)} and {pkg_init.relative_to(out)})"
        )


def _check_python_cli(out: Path, result: PostWriteResult) -> None:
    """Archetype-specific checks for python_cli projects."""
    # Must have some kind of entry point (cli.py, main.py, or __main__.py)
    candidates = [
        out / "src" / "cli.py",
        out / "src" / "main.py",
        out / "main.py",
    ]
    # Also check any package directory for cli.py or __main__.py
    for d in out.iterdir():
        if d.is_dir() and (d / "__init__.py").exists() and d.name not in ("tests", ".venv"):
            candidates.extend([d / "cli.py", d / "__main__.py", d / "main.py"])

    found = [c for c in candidates if c.exists()]
    if found:
        result.checks.append(f"CLI entry point found: {found[0].relative_to(out)}")
    else:
        result.errors.append("No CLI entry point found (cli.py, main.py, or __main__.py)")


def _check_fastapi_service(out: Path, result: PostWriteResult) -> None:
    """Archetype-specific checks for fastapi_service projects."""
    # Must have app.py with a FastAPI app
    candidates = [out / "src" / "app.py"]
    for d in out.iterdir():
        if d.is_dir() and (d / "__init__.py").exists() and d.name not in ("tests", ".venv"):
            candidates.append(d / "app.py")

    found = [c for c in candidates if c.exists()]
    if found:
        content = found[0].read_text()
        if "FastAPI" in content:
            result.checks.append(f"FastAPI app found: {found[0].relative_to(out)}")
        else:
            result.errors.append(f"{found[0].relative_to(out)} exists but doesn't contain FastAPI app")
    else:
        result.errors.append("No app.py with FastAPI found")


def _check_package_import(venv_python: Path, package_name: str, result: PostWriteResult) -> None:
    """Check that the package is importable in the venv."""
    try:
        proc = subprocess.run(
            [str(venv_python), "-c", f"import {package_name}"],
            capture_output=True, text=True, timeout=10,
            cwd=str(venv_python.parent.parent.parent),
        )
        if proc.returncode == 0:
            result.checks.append(f"Package '{package_name}' imports successfully")
        else:
            result.errors.append(f"Package '{package_name}' import failed: {proc.stderr[-200:]}")
    except Exception as e:
        result.errors.append(f"Package '{package_name}' import check failed: {e}")
