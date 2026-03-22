"""Archetype-specific verifier packs.

Deterministic checks that run after the generic verifier. Each pack
knows what a working project of that archetype should look like and
checks for the exact failure patterns we've seen in real runs.

No LLM. Just filesystem + subprocess checks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from build_loop.verifier import SignalResult


def run_archetype_checks(
    output_dir: str,
    archetype: str,
) -> list[SignalResult]:
    """Run archetype-specific checks. Returns signal results."""
    out = Path(output_dir)
    venv_python = out / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    if archetype == "python_cli":
        return _check_python_cli(out, python)
    elif archetype == "fastapi_service":
        return _check_fastapi_service(out, python)
    return []


def _check_python_cli(out: Path, python: str) -> list[SignalResult]:
    """python_cli archetype checks."""
    results = []

    # 1. Find the package directory
    pkg = _find_package_dir(out)
    if not pkg:
        results.append(SignalResult(
            signal_type="archetype", description="Package directory exists",
            passed=False, detail="No Python package directory found (directory with __init__.py)",
        ))
        return results
    results.append(SignalResult(
        signal_type="archetype", description=f"Package directory: {pkg.name}",
        passed=True,
    ))

    # 2. Package importable
    result = _check_import(python, pkg.name, out)
    results.append(result)

    # 3. __init__.py is not a stub
    init = pkg / "__init__.py"
    if init.exists():
        content = init.read_text()
        if len(content.strip()) < 10 or "TODO" in content:
            results.append(SignalResult(
                signal_type="archetype", description="__init__.py has real content",
                passed=False, detail="__init__.py is empty or a stub",
            ))
        else:
            results.append(SignalResult(
                signal_type="archetype", description="__init__.py has real content",
                passed=True,
            ))

    # 4. Entry point resolves to real function (not stub)
    ep_result = _check_entry_point_real(out, pkg.name)
    if ep_result:
        results.append(ep_result)

    # 5. --help exits 0
    help_result = _run_command(
        python, ["-m", pkg.name, "--help"], out,
        "CLI --help exits 0",
    )
    results.append(help_result)

    # 6. No scaffold stubs in package
    stubs = _find_stubs(pkg)
    if stubs:
        results.append(SignalResult(
            signal_type="archetype", description="No scaffold stubs in package",
            passed=False, detail=f"Stubs found: {stubs}",
        ))
    else:
        results.append(SignalResult(
            signal_type="archetype", description="No scaffold stubs in package",
            passed=True,
        ))

    return results


def _check_fastapi_service(out: Path, python: str) -> list[SignalResult]:
    """fastapi_service archetype checks."""
    results = []

    pkg = _find_package_dir(out)
    if not pkg:
        results.append(SignalResult(
            signal_type="archetype", description="Package directory exists",
            passed=False, detail="No Python package directory found",
        ))
        return results
    results.append(SignalResult(
        signal_type="archetype", description=f"Package directory: {pkg.name}",
        passed=True,
    ))

    # 1. Package importable
    results.append(_check_import(python, pkg.name, out))

    # 2. App module exists and contains FastAPI
    app_py = pkg / "app.py"
    if app_py.exists():
        content = app_py.read_text()
        if "FastAPI" in content:
            results.append(SignalResult(
                signal_type="archetype", description="app.py contains FastAPI",
                passed=True,
            ))
        else:
            results.append(SignalResult(
                signal_type="archetype", description="app.py contains FastAPI",
                passed=False, detail="app.py exists but doesn't reference FastAPI",
            ))
    else:
        results.append(SignalResult(
            signal_type="archetype", description="app.py exists",
            passed=False, detail=f"No app.py in {pkg.name}/",
        ))

    # 3. App imports without error
    results.append(_run_command(
        python, ["-c", f"from {pkg.name}.app import app"], out,
        "App module imports successfully",
    ))

    # 4. No scaffold stubs
    stubs = _find_stubs(pkg)
    if stubs:
        results.append(SignalResult(
            signal_type="archetype", description="No scaffold stubs in package",
            passed=False, detail=f"Stubs found: {stubs}",
        ))
    else:
        results.append(SignalResult(
            signal_type="archetype", description="No scaffold stubs in package",
            passed=True,
        ))

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_package_dir(out: Path) -> Path | None:
    """Find the main Python package directory."""
    skip = {"tests", "src", ".venv", "__pycache__", ".build_state", ".git"}
    for d in sorted(out.iterdir()):
        if d.is_dir() and d.name not in skip and (d / "__init__.py").exists():
            return d
    return None


def _check_import(python: str, pkg_name: str, cwd: Path) -> SignalResult:
    """Check that a package is importable."""
    try:
        proc = subprocess.run(
            [python, "-c", f"import {pkg_name}"],
            cwd=str(cwd), capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            return SignalResult(
                signal_type="archetype", description=f"Package '{pkg_name}' importable",
                passed=True,
            )
        return SignalResult(
            signal_type="archetype", description=f"Package '{pkg_name}' importable",
            passed=False, detail=proc.stderr[-300:],
        )
    except Exception as e:
        return SignalResult(
            signal_type="archetype", description=f"Package '{pkg_name}' importable",
            passed=False, detail=str(e),
        )


def _run_command(python: str, args: list[str], cwd: Path, description: str) -> SignalResult:
    """Run a command and check exit code 0."""
    try:
        proc = subprocess.run(
            [python] + args, cwd=str(cwd),
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            return SignalResult(signal_type="archetype", description=description, passed=True)
        return SignalResult(
            signal_type="archetype", description=description,
            passed=False, detail=f"Exit {proc.returncode}: {proc.stderr[-200:]}",
        )
    except Exception as e:
        return SignalResult(
            signal_type="archetype", description=description,
            passed=False, detail=str(e),
        )


def _check_entry_point_real(out: Path, pkg_name: str) -> SignalResult | None:
    """Check that pyproject.toml entry point targets a real function, not a stub."""
    pyproject = out / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        import tomllib
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {})
        for name, target in scripts.items():
            if ":" not in target:
                continue
            module_path, func_name = target.rsplit(":", 1)
            file_path = out / module_path.replace(".", "/")
            py_file = file_path.with_suffix(".py")
            if py_file.exists():
                content = py_file.read_text()
                if "TODO" in content and len(content) < 500:
                    return SignalResult(
                        signal_type="archetype",
                        description=f"Entry point '{name}' targets real code",
                        passed=False,
                        detail=f"{target} points at a scaffold stub",
                    )
                if f"def {func_name}" not in content:
                    return SignalResult(
                        signal_type="archetype",
                        description=f"Entry point '{name}' targets real code",
                        passed=False,
                        detail=f"Function '{func_name}' not found in {py_file.relative_to(out)}",
                    )
                return SignalResult(
                    signal_type="archetype",
                    description=f"Entry point '{name}' targets real code",
                    passed=True,
                )
    except Exception:
        pass
    return None


def _find_stubs(pkg_dir: Path) -> list[str]:
    """Find scaffold stub files in a package directory."""
    MARKERS = ["# TODO", "{{project_name}}", "{{summary}}", "builder fills in", "builder adds"]
    stubs = []
    for py_file in pkg_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text()
        except Exception:
            continue
        if len(content) > 500:
            continue
        for marker in MARKERS:
            if marker in content:
                stubs.append(str(py_file.relative_to(pkg_dir)))
                break
    return stubs
