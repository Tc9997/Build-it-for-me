"""Eval-owned verification: runs the corpus's expected_signals against the output.

This is the eval harness's own scoring — independent of the build system's
internal verifier. Both modes are scored by the same benchmark assertions.

All commands run against the project's own .venv if it exists, not the
host interpreter. This ensures eval results reflect what the build produced,
not what happens to be installed on the host.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from build_loop.eval.models import EvalTask


def _resolve_python(output_dir: Path) -> str:
    """Return the project's venv python if it exists, else sys.executable."""
    venv_python = output_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _resolve_command(command: str, output_dir: Path) -> str:
    """Resolve a command name to the project's venv if applicable.

    'python' → .venv/bin/python (if venv exists)
    'pytest' → .venv/bin/pytest (if venv exists)
    Other commands are returned as-is.
    """
    if command in ("python", "python3", "pytest", "pip"):
        venv_bin = output_dir / ".venv" / "bin" / command
        if venv_bin.exists():
            return str(venv_bin)
    return command


def _project_env(output_dir: Path) -> dict[str, str]:
    """Build an environment dict that uses the project's venv.

    Sets PATH to prefer .venv/bin and PYTHONPATH to include the project root.
    """
    env = os.environ.copy()
    venv_bin = output_dir / ".venv" / "bin"
    if venv_bin.exists():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(output_dir / ".venv")
    env["PYTHONPATH"] = str(output_dir)
    return env


def eval_verify(task: EvalTask, output_dir: Path) -> list[dict]:
    """Run the corpus's expected_signals against the built output.

    Commands are resolved against the project's .venv, not the host.
    """
    results = []
    env = _project_env(output_dir)

    for sig in task.expected_signals:
        sig_type = sig.get("type", "")
        description = sig.get("description", "")

        if sig_type == "cli_exit":
            result = _check_cli_exit(sig, output_dir, env)
        elif sig_type == "file_exists":
            result = _check_file_exists(sig, output_dir)
        elif sig_type == "import_check":
            result = _check_import(sig, output_dir, env)
        elif sig_type == "stdout_contains":
            result = _check_stdout_contains(sig, output_dir, env)
        else:
            result = {"passed": False, "detail": f"Unknown signal type: {sig_type}"}

        result["type"] = sig_type
        result["description"] = description
        results.append(result)

    return results


def _check_cli_exit(sig: dict, output_dir: Path, env: dict) -> dict:
    command = _resolve_command(sig.get("command", ""), output_dir)
    args = sig.get("args", [])
    expect_exit = sig.get("expect_exit", 0)

    argv = [command] + args
    try:
        proc = subprocess.run(
            argv, cwd=str(output_dir), env=env,
            capture_output=True, text=True, timeout=30,
        )
        passed = proc.returncode == expect_exit
        detail = "" if passed else (
            f"Expected exit {expect_exit}, got {proc.returncode}. "
            f"stderr: {proc.stderr[-300:]}"
        )
    except FileNotFoundError:
        passed, detail = False, f"Command not found: {command}"
    except subprocess.TimeoutExpired:
        passed, detail = False, "Timed out"

    return {"passed": passed, "detail": detail}


def _check_file_exists(sig: dict, output_dir: Path) -> dict:
    file_path = sig.get("file_path", "")
    target = output_dir / file_path
    passed = target.exists()
    detail = "" if passed else f"Not found: {file_path}"
    return {"passed": passed, "detail": detail}


def _check_import(sig: dict, output_dir: Path, env: dict) -> dict:
    module_name = sig.get("module_name", "")
    python = _resolve_python(output_dir)
    try:
        proc = subprocess.run(
            [python, "-c", f"import {module_name}"],
            cwd=str(output_dir), env=env,
            capture_output=True, text=True, timeout=10,
        )
        passed = proc.returncode == 0
        detail = "" if passed else proc.stderr[-300:]
    except Exception as e:
        passed, detail = False, str(e)
    return {"passed": passed, "detail": detail}


def _check_stdout_contains(sig: dict, output_dir: Path, env: dict) -> dict:
    command = _resolve_command(sig.get("command", ""), output_dir)
    args = sig.get("args", [])
    expect_contains = sig.get("expect_contains", "")

    argv = [command] + args
    try:
        proc = subprocess.run(
            argv, cwd=str(output_dir), env=env,
            capture_output=True, text=True, timeout=30,
        )
        passed = expect_contains in proc.stdout
        detail = "" if passed else f"Expected '{expect_contains}' in stdout"
    except FileNotFoundError:
        passed, detail = False, f"Command not found: {command}"
    except subprocess.TimeoutExpired:
        passed, detail = False, "Timed out"

    return {"passed": passed, "detail": detail}
