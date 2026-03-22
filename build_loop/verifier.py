"""Independent verifier: executes contract signals and deterministic checks.

The verifier is the authority for pass/fail. It does NOT rely on builder-
generated tests. It does NOT invent checks from prose. It executes:

Tier 1 (deterministic, always):
  - Syntax check (py_compile) on all .py files
  - Import check for signals of type import_check

Tier 2 (contract-derived):
  - Execute every SuccessSignal from the BuildContract
  - cli_exit, stdout_contains, file_exists, http_probe, import_check, schema_valid

Behavioral expectations and invariants that are not structurally executable
are reported as UNCOVERED — not pretended to be verified.

For service-mode projects with http_probe signals, the verifier starts the
service process, waits for it to be ready, runs the probes, then terminates.
This reuses the same liveness logic as the executor's service smoke test.
"""

from __future__ import annotations

import json
import py_compile
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console

from build_loop.contract import (
    BuildContract,
    CliExitSignal,
    FileExistsSignal,
    HttpProbeSignal,
    ImportCheckSignal,
    SchemaValidSignal,
    StdoutContainsSignal,
)
from build_loop.safety import PathTraversalError, safe_command, safe_output_path, UnsafeCommandError
from build_loop.schemas import ExecResult

console = Console()

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class SignalResult(BaseModel):
    """Result of executing a single SuccessSignal."""
    signal_type: str
    description: str
    passed: bool
    detail: str = ""


class VerificationResult(BaseModel):
    """Full output of the verifier. Persisted in state. Consumed by acceptance."""
    schema_version: str = SCHEMA_VERSION
    tier1_passed: bool = True
    tier1_results: list[SignalResult] = Field(default_factory=list)
    tier2_passed: bool = True
    tier2_results: list[SignalResult] = Field(default_factory=list)
    tier3_passed: bool = True
    tier3_results: list[SignalResult] = Field(default_factory=list)
    uncovered_behavioral: list[str] = Field(
        default_factory=list,
        description="Behavioral expectations not structurally executable"
    )
    uncovered_invariants: list[str] = Field(
        default_factory=list,
        description="Invariants not structurally executable"
    )

    @property
    def passed(self) -> bool:
        return self.tier1_passed and self.tier2_passed and self.tier3_passed

    @property
    def summary(self) -> str:
        t1 = sum(1 for r in self.tier1_results if r.passed)
        t2 = sum(1 for r in self.tier2_results if r.passed)
        t3 = sum(1 for r in self.tier3_results if r.passed)
        total_pass = t1 + t2 + t3
        total = len(self.tier1_results) + len(self.tier2_results) + len(self.tier3_results)
        uncovered = len(self.uncovered_behavioral) + len(self.uncovered_invariants)
        return (
            f"{total_pass}/{total} checks passed"
            + (f", {uncovered} uncovered" if uncovered else "")
        )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """Independent verification harness. No LLM. Executes contract signals."""

    name = "verifier"

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)

    def log(self, msg: str) -> None:
        console.print(f"[bold blue][{self.name}][/bold blue] {msg}")

    def run(
        self,
        contract: BuildContract,
        run_command: str | None = None,
    ) -> VerificationResult:
        """Execute all verification tiers against the contract.

        Args:
            contract: The BuildContract with signals to execute.
            run_command: For service-mode projects, the command to start the
                service. Required if contract has http_probe signals and
                run_mode is "service".
        """
        result = VerificationResult()

        # Tier 1: deterministic checks
        self.log("Tier 1: deterministic checks...")
        self._tier1_syntax(result)
        self._tier1_import_checks(result, contract)

        result.tier1_passed = all(r.passed for r in result.tier1_results)
        t1_pass = sum(1 for r in result.tier1_results if r.passed)
        self.log(f"  Tier 1: {t1_pass}/{len(result.tier1_results)} passed")

        # Tier 2: contract signal execution
        # For service-mode with http_probe signals, manage the service lifecycle
        has_http_probes = any(isinstance(s, HttpProbeSignal) for s in contract.success_signals)
        service_proc = None

        if contract.run_mode == "service" and has_http_probes and run_command:
            # Use the first http_probe URL as the readiness endpoint
            first_probe = next(
                (s for s in contract.success_signals if isinstance(s, HttpProbeSignal)), None
            )
            probe_url = None
            if first_probe:
                probe_url = first_probe.path
                if not probe_url.startswith("http"):
                    probe_url = f"http://localhost:8000{probe_url}"
            service_proc = self._start_service(run_command, probe_url=probe_url)
            if service_proc is None:
                result.tier2_results.append(SignalResult(
                    signal_type="service_start", description="Start service for http_probe",
                    passed=False, detail=f"Failed to start service: {run_command}",
                ))

        try:
            self.log("Tier 2: contract signals...")
            for signal in contract.success_signals:
                sr = self._execute_signal(signal, contract.run_mode)
                result.tier2_results.append(sr)
                status = "[green]PASS[/green]" if sr.passed else "[red]FAIL[/red]"
                self.log(f"  {status} [{sr.signal_type}] {sr.description}")
                if not sr.passed and sr.detail:
                    self.log(f"    {sr.detail}")
        finally:
            if service_proc is not None:
                self._stop_service(service_proc)

        result.tier2_passed = all(r.passed for r in result.tier2_results)

        # Tier 3: archetype-specific checks
        from build_loop.analysis.archetype_checks import run_archetype_checks
        self.log("Tier 3: archetype checks...")
        tier3 = run_archetype_checks(str(self.project_dir), contract.archetype)
        result.tier3_results = tier3
        result.tier3_passed = all(r.passed for r in tier3)
        for sr in tier3:
            status = "[green]PASS[/green]" if sr.passed else "[red]FAIL[/red]"
            self.log(f"  {status} [{sr.signal_type}] {sr.description}")
            if not sr.passed and sr.detail:
                self.log(f"    {sr.detail}")
        t3_pass = sum(1 for r in tier3 if r.passed)
        self.log(f"  Tier 3: {t3_pass}/{len(tier3)} passed")

        # Report uncovered items honestly
        for be in contract.behavioral_expectations:
            result.uncovered_behavioral.append(
                f"{be.description} (given: {be.given}, expect: {be.expect})"
            )
        for inv in contract.invariants:
            result.uncovered_invariants.append(
                f"[{inv.category}] {inv.description}"
            )

        if result.uncovered_behavioral:
            self.log(f"  {len(result.uncovered_behavioral)} behavioral expectations uncovered")
        if result.uncovered_invariants:
            self.log(f"  {len(result.uncovered_invariants)} invariants uncovered")

        self.log(f"  Result: {'PASS' if result.passed else 'FAIL'} — {result.summary}")
        return result

    # ------------------------------------------------------------------
    # Tier 1
    # ------------------------------------------------------------------

    def _tier1_syntax(self, result: VerificationResult) -> None:
        """Syntax-check all Python files in the project."""
        py_files = list(self.project_dir.rglob("*.py"))
        py_files = [
            f for f in py_files
            if not any(skip in str(f) for skip in [".venv", "__pycache__", ".build_state"])
        ]

        for f in py_files:
            rel = str(f.relative_to(self.project_dir))
            try:
                py_compile.compile(str(f), doraise=True)
                result.tier1_results.append(SignalResult(
                    signal_type="syntax", description=f"Syntax OK: {rel}", passed=True,
                ))
            except py_compile.PyCompileError as e:
                result.tier1_results.append(SignalResult(
                    signal_type="syntax", description=f"Syntax error: {rel}",
                    passed=False, detail=str(e),
                ))

    def _tier1_import_checks(self, result: VerificationResult, contract: BuildContract) -> None:
        """Run import_check signals as tier 1 (they're deterministic)."""
        for signal in contract.success_signals:
            if isinstance(signal, ImportCheckSignal):
                sr = self._check_import(signal)
                result.tier1_results.append(sr)

    # ------------------------------------------------------------------
    # Tier 2: signal execution
    # ------------------------------------------------------------------

    def _execute_signal(self, signal, run_mode: str) -> SignalResult:
        """Dispatch a signal to its executor."""
        if isinstance(signal, CliExitSignal):
            return self._check_cli_exit(signal)
        elif isinstance(signal, StdoutContainsSignal):
            return self._check_stdout_contains(signal)
        elif isinstance(signal, HttpProbeSignal):
            return self._check_http_probe(signal, run_mode)
        elif isinstance(signal, FileExistsSignal):
            return self._check_file_exists(signal)
        elif isinstance(signal, ImportCheckSignal):
            # Already handled in tier 1, but report pass here too
            return self._check_import(signal)
        elif isinstance(signal, SchemaValidSignal):
            return self._check_schema_valid(signal)
        else:
            return SignalResult(
                signal_type="unknown", description=f"Unknown signal type",
                passed=False, detail=f"Cannot execute: {signal}",
            )

    @staticmethod
    def _normalize_argv(command: str, args: list[str]) -> list[str]:
        """Normalize a command + args into a proper argv list."""
        if " " in command and not args:
            return shlex.split(command)
        return [command] + args

    def _safe_execute(self, command: str, args: list[str], description: str, timeout: int = 60) -> tuple[subprocess.CompletedProcess | None, str | None]:
        """Validate and execute a command safely. Returns (proc, error_detail).

        Rejects:
        - Absolute paths as command (except .venv/bin/ executables)
        - python -c (arbitrary code execution)
        - python -m pip (package manipulation)
        - Non-python/non-venv executables

        Allows:
        - python -m <project_module>
        - python <project-relative-script.py>
        - .venv/bin/<tool> (installed entry points)
        """
        argv = self._normalize_argv(command, args)
        if not argv:
            return None, "Empty command"

        executable = argv[0]
        venv_bin = self.project_dir / ".venv" / "bin"

        # Allow .venv/bin/ executables (installed entry points)
        if str(executable).startswith(str(venv_bin)):
            pass  # Trusted — installed by pip into project venv
        elif executable in ("python", "python3") or executable == sys.executable:
            # Validate python subcommands
            if len(argv) > 1:
                flag = argv[1]
                if flag == "-c":
                    return None, "Rejected: python -c is not allowed (arbitrary code execution)"
                if flag == "-m" and len(argv) > 2 and argv[2] == "pip":
                    return None, "Rejected: python -m pip is not allowed in verifier"
                if flag == "-m":
                    pass  # python -m <module> is ok
                elif flag.startswith("/"):
                    return None, f"Rejected: absolute script path {flag}"
                elif ".." in flag:
                    return None, f"Rejected: path traversal in script path {flag}"
                # else: python <relative-script.py> — ok
            # Check all args for path traversal
            for arg in argv[1:]:
                if ".." in arg and not arg.startswith("-"):
                    return None, f"Rejected: path traversal in argument: {arg}"
        elif executable.startswith("/"):
            return None, f"Rejected: absolute executable path {executable}"
        else:
            return None, f"Rejected: unknown executable {executable}. Only python and .venv/bin/ tools are allowed."

        try:
            proc = subprocess.run(
                argv, cwd=str(self.project_dir),
                capture_output=True, text=True, timeout=timeout,
            )
            return proc, None
        except FileNotFoundError:
            return None, f"Command not found: {executable}"
        except subprocess.TimeoutExpired:
            return None, "Command timed out"

    def _check_cli_exit(self, signal: CliExitSignal) -> SignalResult:
        """Run a command and check exit code."""
        proc, error = self._safe_execute(signal.command, signal.args, signal.description)
        if error:
            return SignalResult(
                signal_type="cli_exit", description=signal.description,
                passed=False, detail=error,
            )
        passed = proc.returncode == signal.expect_exit
        detail = "" if passed else (
            f"Expected exit {signal.expect_exit}, got {proc.returncode}. "
            f"stderr: {proc.stderr[-500:]}"
        )
        return SignalResult(
            signal_type="cli_exit", description=signal.description,
            passed=passed, detail=detail,
        )

    def _check_stdout_contains(self, signal: StdoutContainsSignal) -> SignalResult:
        """Run a command and check stdout contains expected string."""
        proc, error = self._safe_execute(signal.command, signal.args, signal.description)
        if error:
            return SignalResult(
                signal_type="stdout_contains", description=signal.description,
                passed=False, detail=error,
            )
        passed = signal.expect_contains in proc.stdout
        detail = "" if passed else (
            f"Expected stdout to contain '{signal.expect_contains}'. "
            f"Got: {proc.stdout[-500:]}"
        )
        return SignalResult(
            signal_type="stdout_contains", description=signal.description,
            passed=passed, detail=detail,
        )

    def _check_http_probe(self, signal: HttpProbeSignal, run_mode: str) -> SignalResult:
        """Probe an HTTP endpoint. Only HTTP/HTTPS or relative paths allowed.

        Relative paths are allowed (prepended with localhost for the project's
        own service). Absolute URLs are checked for SSRF — no loopback/private.
        """
        url = signal.path
        if url.startswith("/"):
            # Relative path — prepend localhost (probing the project's own service)
            url = f"http://localhost:8000{signal.path}"
        elif url.startswith("http://") or url.startswith("https://"):
            # Absolute URL — check for SSRF (no loopback/private except localhost:8000)
            from build_loop.web import _is_blocked_host
            blocked = _is_blocked_host(url)
            if blocked:
                return SignalResult(
                    signal_type="http_probe", description=signal.description,
                    passed=False, detail=f"Rejected: {blocked}",
                )
        else:
            return SignalResult(
                signal_type="http_probe", description=signal.description,
                passed=False,
                detail=f"Rejected: URL must be http://, https://, or a relative path. Got: {url}",
            )

        try:
            curl_argv = [
                "curl", "-s", "-o", "/dev/null",
                "-w", "%{http_code}",
                "-X", signal.method,
                "--max-time", "10",
                url,
            ]
            proc = subprocess.run(
                curl_argv, capture_output=True, text=True, timeout=15,
            )
            status_code = int(proc.stdout.strip()) if proc.stdout.strip().isdigit() else 0
            passed = status_code == signal.expect_status

            if passed and signal.expect_body_contains:
                # Re-fetch with body
                body_proc = subprocess.run(
                    ["curl", "-s", "-X", signal.method, "--max-time", "10", url],
                    capture_output=True, text=True, timeout=15,
                )
                passed = signal.expect_body_contains in body_proc.stdout
                if not passed:
                    return SignalResult(
                        signal_type="http_probe", description=signal.description,
                        passed=False,
                        detail=f"Status {status_code} OK but body missing '{signal.expect_body_contains}'",
                    )

            detail = "" if passed else f"Expected status {signal.expect_status}, got {status_code}"
        except Exception as e:
            passed, detail = False, f"HTTP probe failed: {e}"

        return SignalResult(
            signal_type="http_probe", description=signal.description,
            passed=passed, detail=detail,
        )

    def _check_file_exists(self, signal: FileExistsSignal) -> SignalResult:
        """Check that a file exists in the project directory.

        Uses safe_output_path to reject absolute paths and .. traversal.
        """
        try:
            resolved = safe_output_path(self.project_dir, signal.file_path)
            passed = resolved.exists()
            detail = "" if passed else f"File not found: {signal.file_path}"
        except PathTraversalError as e:
            passed = False
            detail = f"Path rejected: {e}"
        return SignalResult(
            signal_type="file_exists", description=signal.description,
            passed=passed, detail=detail,
        )

    def _check_import(self, signal: ImportCheckSignal) -> SignalResult:
        """Check that a Python module is importable in the project venv.

        Uses python -c directly (not _safe_execute) because import checks
        require -c. Module name is validated to prevent injection.
        """
        import re as _re
        # Validate module_name is a safe Python identifier path
        if not _re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", signal.module_name):
            return SignalResult(
                signal_type="import_check", description=signal.description,
                passed=False,
                detail=f"Rejected: invalid module name '{signal.module_name}'",
            )

        venv_python = self.project_dir / ".venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else sys.executable
        try:
            proc = subprocess.run(
                [python, "-c", f"import {signal.module_name}"],
                cwd=str(self.project_dir),
                capture_output=True, text=True, timeout=15,
            )
            passed = proc.returncode == 0
            detail = "" if passed else proc.stderr[-500:]
        except Exception as e:
            passed, detail = False, str(e)

        return SignalResult(
            signal_type="import_check", description=signal.description,
            passed=passed, detail=detail,
        )

    # ------------------------------------------------------------------
    # Service lifecycle (for http_probe on service-mode projects)
    # ------------------------------------------------------------------

    def _start_service(
        self,
        run_command: str,
        probe_url: str | None = None,
        timeout: float = 15.0,
        poll_interval: float = 0.3,
    ) -> subprocess.Popen | None:
        """Start a service process and wait until it's ready.

        Readiness is determined by:
          1. If probe_url is given: poll it until it returns any HTTP response.
          2. Otherwise: wait poll_interval, check process is still alive.

        Returns the Popen handle, or None if the service failed to start
        or never became ready within timeout.
        """
        try:
            argv = safe_command(run_command)
        except UnsafeCommandError as e:
            self.log(f"  [red]Service command rejected: {e}[/red]")
            return None

        self.log(f"  Starting service: {run_command}")
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            self.log(f"  [red]Service command not found: {argv[0]}[/red]")
            return None

        if probe_url:
            # Readiness-based: poll the endpoint until it responds
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    self.log(
                        f"  [red]Service exited during startup (exit {proc.returncode})[/red]"
                        + (f"\n    stderr: {stderr[-500:]}" if stderr else "")
                    )
                    return None
                try:
                    r = subprocess.run(
                        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                         "--max-time", "2", probe_url],
                        capture_output=True, text=True, timeout=5,
                    )
                    if r.stdout.strip().isdigit() and int(r.stdout.strip()) > 0:
                        self.log(f"  [green]Service ready (responded at {probe_url})[/green]")
                        return proc
                except Exception:
                    pass
                time.sleep(poll_interval)

            # Timeout — check if process is at least alive
            if proc.poll() is None:
                self.log(f"  [yellow]Service alive but never responded at {probe_url} within {timeout}s[/yellow]")
                return proc  # Let probes try anyway
            else:
                self.log(f"  [red]Service exited and never became ready[/red]")
                return None
        else:
            # No probe URL — basic liveness check after a short wait
            time.sleep(min(1.0, timeout))
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                self.log(
                    f"  [red]Service exited during startup (exit {proc.returncode})[/red]"
                    + (f"\n    stderr: {stderr[-500:]}" if stderr else "")
                )
                return None
            self.log(f"  [green]Service alive[/green]")
            return proc

    def _stop_service(self, proc: subprocess.Popen) -> None:
        """Gracefully terminate a service process."""
        self.log("  Stopping service...")
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()

    def _check_schema_valid(self, signal: SchemaValidSignal) -> SignalResult:
        """Run a command, parse stdout as JSON, validate against schema."""
        proc, error = self._safe_execute(signal.command, signal.args, signal.description)
        if error:
            return SignalResult(
                signal_type="schema_valid", description=signal.description,
                passed=False, detail=error,
            )
        try:
            if proc.returncode != 0:
                return SignalResult(
                    signal_type="schema_valid", description=signal.description,
                    passed=False, detail=f"Command failed with exit {proc.returncode}",
                )
            data = json.loads(proc.stdout)
            # Basic JSON schema validation via jsonschema if available,
            # otherwise just check that output is valid JSON
            try:
                import jsonschema
                jsonschema.validate(data, signal.json_schema)
                passed, detail = True, ""
            except ImportError:
                # Cannot validate schema without jsonschema library — fail honestly
                passed, detail = False, (
                    "jsonschema library not installed — cannot validate JSON schema. "
                    "Install with: pip install jsonschema"
                )
            except jsonschema.ValidationError as e:
                passed, detail = False, f"Schema validation failed: {e.message}"
        except json.JSONDecodeError as e:
            passed, detail = False, f"Output is not valid JSON: {e}"
        except Exception as e:
            passed, detail = False, str(e)

        return SignalResult(
            signal_type="schema_valid", description=signal.description,
            passed=passed, detail=detail,
        )
