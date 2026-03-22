"""Eval-owned verification: runs the corpus's expected_signals against the output.

This is the eval harness's own scoring — independent of the build system's
internal verifier. Both modes are scored by the same benchmark assertions.
The build system's self-reported verification is recorded but does NOT
determine the eval pass/fail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from build_loop.eval.models import EvalTask


def eval_verify(task: EvalTask, output_dir: Path) -> list[dict]:
    """Run the corpus's expected_signals against the built output.

    Returns a list of signal results, each with:
      type, description, passed, detail
    """
    results = []
    for sig in task.expected_signals:
        sig_type = sig.get("type", "")
        description = sig.get("description", "")

        if sig_type == "cli_exit":
            result = _check_cli_exit(sig, output_dir)
        elif sig_type == "file_exists":
            result = _check_file_exists(sig, output_dir)
        elif sig_type == "import_check":
            result = _check_import(sig, output_dir)
        elif sig_type == "stdout_contains":
            result = _check_stdout_contains(sig, output_dir)
        else:
            result = {"passed": False, "detail": f"Unknown signal type: {sig_type}"}

        result["type"] = sig_type
        result["description"] = description
        results.append(result)

    return results


def _check_cli_exit(sig: dict, output_dir: Path) -> dict:
    command = sig.get("command", "")
    args = sig.get("args", [])
    expect_exit = sig.get("expect_exit", 0)

    argv = [command] + args
    try:
        proc = subprocess.run(
            argv, cwd=str(output_dir),
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


def _check_import(sig: dict, output_dir: Path) -> dict:
    module_name = sig.get("module_name", "")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            cwd=str(output_dir),
            capture_output=True, text=True, timeout=10,
        )
        passed = proc.returncode == 0
        detail = "" if passed else proc.stderr[-300:]
    except Exception as e:
        passed, detail = False, str(e)
    return {"passed": passed, "detail": detail}


def _check_stdout_contains(sig: dict, output_dir: Path) -> dict:
    command = sig.get("command", "")
    args = sig.get("args", [])
    expect_contains = sig.get("expect_contains", "")

    argv = [command] + args
    try:
        proc = subprocess.run(
            argv, cwd=str(output_dir),
            capture_output=True, text=True, timeout=30,
        )
        passed = expect_contains in proc.stdout
        detail = "" if passed else f"Expected '{expect_contains}' in stdout"
    except FileNotFoundError:
        passed, detail = False, f"Command not found: {command}"
    except subprocess.TimeoutExpired:
        passed, detail = False, "Timed out"

    return {"passed": passed, "detail": detail}
