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
    export_metadata: dict | None = None,
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

    # 3. Detect scaffold stubs that survived into final output
    _check_surviving_scaffolds(out, result)

    # 4. Archetype-specific checks
    if archetype == "python_cli":
        _check_python_cli(out, result)
    elif archetype == "fastapi_service":
        _check_fastapi_service(out, result)

    # 5. README code example validation
    if export_metadata:
        from build_loop.analysis.readme_validation import validate_readme
        readme_errors = validate_readme(out, export_metadata)
        for err in readme_errors:
            result.errors.append(f"README: {err}")

    # NOTE: Package importability is checked in run_post_setup_checks(),
    # not here. The venv doesn't exist yet at post-write time.

    if result.errors:
        result.passed = False

    return result


def run_post_setup_checks(output_dir: str) -> PostWriteResult:
    """Run checks that require the venv to exist (after SETUP phase).

    Separated from post-write checks because the venv doesn't exist
    at post-write time.
    """
    result = PostWriteResult()
    out = Path(output_dir)

    venv_python = out / ".venv" / "bin" / "python"
    if not venv_python.exists():
        result.checks.append("No venv found — skipping post-setup checks")
        return result

    # Package importable
    pkg_dirs = [d for d in out.iterdir()
                if d.is_dir() and (d / "__init__.py").exists()
                and d.name not in ("tests", "src", ".venv", "__pycache__", ".git")]
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
        content = py_file.read_text()
        if "# TODO" in content and len(content) < 500:
            result.errors.append(
                f"Entry point '{name}': {py_file.relative_to(out)} is a scaffold stub with TODO"
            )
        elif f"def {func_name}" in content:
            result.checks.append(f"Entry point '{name}': {target} — found")
        else:
            result.errors.append(
                f"Entry point '{name}': function '{func_name}' not found in {py_file.relative_to(out)}"
            )
    elif pkg_init.exists():
        # Package exists — check that the function is defined in __init__.py
        init_content = pkg_init.read_text()
        if f"def {func_name}" in init_content:
            result.checks.append(f"Entry point '{name}': {target} — found in __init__.py")
        else:
            result.errors.append(
                f"Entry point '{name}': function '{func_name}' not found in "
                f"{pkg_init.relative_to(out)}. The entry point targets the package "
                f"but '{func_name}' is not defined or imported there."
            )
    else:
        result.errors.append(
            f"Entry point '{name}': module '{module_path}' not found "
            f"(checked {py_file.relative_to(out)} and {pkg_init.relative_to(out)})"
        )


def _check_python_cli(out: Path, result: PostWriteResult) -> None:
    """Archetype-specific checks for python_cli projects."""
    # A valid [project.scripts] entry already proves there's a CLI entry point.
    # Only check for file-level candidates if no console_scripts entry was validated.
    if any("Entry point" in c and "found" in c for c in result.checks):
        return  # Already validated by _check_entry_point

    candidates = [out / "main.py"]
    for d in out.iterdir():
        if d.is_dir() and (d / "__init__.py").exists() and d.name not in ("tests", ".venv", "__pycache__"):
            candidates.extend([d / "cli.py", d / "__main__.py", d / "main.py"])

    found = [c for c in candidates if c.exists()]
    if found:
        result.checks.append(f"CLI entry point found: {found[0].relative_to(out)}")
    else:
        result.errors.append("No CLI entry point found (cli.py, main.py, __main__.py, or [project.scripts])")


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


def _check_surviving_scaffolds(out: Path, result: PostWriteResult) -> None:
    """Detect template scaffold stubs that weren't overwritten by builders.

    Scaffold files are short (.py < 500 chars) with TODO comments,
    placeholder text like '{{project_name}}', or 'builder fills in'.
    """
    SCAFFOLD_MARKERS = ["# TODO", "{{project_name}}", "{{summary}}", "builder fills in", "builder adds"]
    for py_file in out.rglob("*.py"):
        if any(skip in str(py_file) for skip in [".venv", "__pycache__", ".build_state"]):
            continue
        try:
            content = py_file.read_text()
        except Exception:
            continue
        if len(content) > 500:
            continue
        for marker in SCAFFOLD_MARKERS:
            if marker in content:
                rel = str(py_file.relative_to(out))
                result.errors.append(
                    f"Scaffold stub survived: {rel} contains '{marker}'"
                )
                break
