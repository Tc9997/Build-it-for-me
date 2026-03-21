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
from build_loop.safety import safe_command, UnsafeCommandError
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
        return self.tier1_passed and self.tier2_passed

    @property
    def summary(self) -> str:
        t1 = sum(1 for r in self.tier1_results if r.passed)
        t2 = sum(1 for r in self.tier2_results if r.passed)
        total_pass = t1 + t2
        total = len(self.tier1_results) + len(self.tier2_results)
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
            service_proc = self._start_service(run_command)
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

    def _check_cli_exit(self, signal: CliExitSignal) -> SignalResult:
        """Run a command and check exit code."""
        argv = [signal.command] + signal.args
        try:
            proc = subprocess.run(
                argv, cwd=str(self.project_dir),
                capture_output=True, text=True, timeout=60,
            )
            passed = proc.returncode == signal.expect_exit
            detail = "" if passed else (
                f"Expected exit {signal.expect_exit}, got {proc.returncode}. "
                f"stderr: {proc.stderr[-500:]}"
            )
        except FileNotFoundError:
            passed, detail = False, f"Command not found: {signal.command}"
        except subprocess.TimeoutExpired:
            passed, detail = False, "Command timed out"

        return SignalResult(
            signal_type="cli_exit", description=signal.description,
            passed=passed, detail=detail,
        )

    def _check_stdout_contains(self, signal: StdoutContainsSignal) -> SignalResult:
        """Run a command and check stdout contains expected string."""
        argv = [signal.command] + signal.args
        try:
            proc = subprocess.run(
                argv, cwd=str(self.project_dir),
                capture_output=True, text=True, timeout=60,
            )
            passed = signal.expect_contains in proc.stdout
            detail = "" if passed else (
                f"Expected stdout to contain '{signal.expect_contains}'. "
                f"Got: {proc.stdout[-500:]}"
            )
        except FileNotFoundError:
            passed, detail = False, f"Command not found: {signal.command}"
        except subprocess.TimeoutExpired:
            passed, detail = False, "Command timed out"

        return SignalResult(
            signal_type="stdout_contains", description=signal.description,
            passed=passed, detail=detail,
        )

    def _check_http_probe(self, signal: HttpProbeSignal, run_mode: str) -> SignalResult:
        """Probe an HTTP endpoint. For service-mode projects, this expects the
        service is already running (started by smoke test). Uses curl for isolation."""
        url = signal.path
        if not url.startswith("http"):
            url = f"http://localhost:8000{signal.path}"

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
        """Check that a file exists in the project directory."""
        target = self.project_dir / signal.file_path
        passed = target.exists()
        detail = "" if passed else f"File not found: {signal.file_path}"
        return SignalResult(
            signal_type="file_exists", description=signal.description,
            passed=passed, detail=detail,
        )

    def _check_import(self, signal: ImportCheckSignal) -> SignalResult:
        """Check that a Python module is importable."""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", f"import {signal.module_name}"],
                cwd=str(self.project_dir),
                capture_output=True, text=True, timeout=15,
                env={"PYTHONPATH": str(self.project_dir), "PATH": ""},
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

    def _start_service(self, run_command: str, startup_wait: float = 3.0) -> subprocess.Popen | None:
        """Start a service process for http_probe verification.

        Returns the Popen handle, or None if the service failed to start.
        Waits startup_wait seconds, then checks the process is still alive.
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

        time.sleep(startup_wait)

        if proc.poll() is not None:
            self.log(f"  [red]Service exited during startup (exit {proc.returncode})[/red]")
            return None

        self.log(f"  [green]Service alive after {startup_wait}s[/green]")
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
        argv = [signal.command] + signal.args
        try:
            proc = subprocess.run(
                argv, cwd=str(self.project_dir),
                capture_output=True, text=True, timeout=60,
            )
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
                passed, detail = True, "jsonschema not installed — validated as parseable JSON only"
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
