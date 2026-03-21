"""Executor agent: runs commands in the project directory and captures results.

All commands are validated through safety.safe_command() before execution.
No shell=True anywhere — every command runs as a parsed argv list.
"""

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path

from rich.console import Console

from build_loop.safety import UnsafeCommandError, safe_command
from build_loop.schemas import ExecResult

console = Console()

DEFAULT_TIMEOUT = 120  # seconds
SMOKE_PROBE_WINDOW = 5  # seconds to wait before checking if a service is alive


class ExecutorAgent:
    """Not an LLM agent — this one actually runs code."""

    name = "executor"

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        console.print(f"[bold magenta][{self.name}][/bold magenta] {msg}")

    def run_command(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> ExecResult:
        """Validate and run a command in the project directory. No shell."""
        self.log(f"$ {command}")

        try:
            argv = safe_command(command)
        except UnsafeCommandError as e:
            self.log(f"  [bold red]REJECTED: {e}[/bold red]")
            return ExecResult(
                command=command,
                exit_code=-2,
                stderr=f"Command rejected by safety check: {e}",
            )

        try:
            proc = subprocess.run(
                argv,
                shell=False,
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result = ExecResult(
                command=command,
                exit_code=proc.returncode,
                stdout=proc.stdout[-5000:] if len(proc.stdout) > 5000 else proc.stdout,
                stderr=proc.stderr[-5000:] if len(proc.stderr) > 5000 else proc.stderr,
            )
        except FileNotFoundError:
            result = ExecResult(
                command=command,
                exit_code=-1,
                stderr=f"Command not found: {argv[0]}",
            )
        except subprocess.TimeoutExpired:
            result = ExecResult(
                command=command,
                exit_code=-1,
                stderr=f"Command timed out after {timeout}s",
                timed_out=True,
            )

        if result.success:
            self.log(f"  [green]OK[/green] (exit 0)")
        else:
            self.log(f"  [red]FAILED[/red] (exit {result.exit_code})")
            if result.stderr:
                lines = result.stderr.strip().split("\n")
                for line in lines[-10:]:
                    self.log(f"  [red]{line}[/red]")

        return result

    def setup_project(self, commands: list[str]) -> list[ExecResult]:
        """Run a sequence of setup commands. Stops on first failure."""
        results = []
        for cmd in commands:
            result = self.run_command(cmd)
            results.append(result)
            if not result.success:
                self.log(f"[red]Setup failed at: {cmd}[/red]")
                break
        return results

    def run_tests(self, test_command: str = "pytest -v") -> ExecResult:
        """Run the test suite."""
        self.log("running tests...")
        return self.run_command(test_command, timeout=180)

    def smoke_test(self, run_command: str, run_mode: str = "batch", timeout: int = 30) -> ExecResult:
        """Run the project and check if it works.

        For run_mode="batch": run to completion and check exit code (existing behavior).
        For run_mode="service": start the process, wait a probe window, and check
        that it's still alive (healthy for daemons/servers), then terminate it.
        """
        self.log(f"smoke test ({run_mode}): {run_command}")

        try:
            argv = safe_command(run_command)
        except UnsafeCommandError as e:
            self.log(f"  [bold red]REJECTED: {e}[/bold red]")
            return ExecResult(
                command=run_command,
                exit_code=-2,
                stderr=f"Command rejected by safety check: {e}",
            )

        if run_mode == "service":
            return self._smoke_test_service(argv, run_command, timeout)
        else:
            return self.run_command(run_command, timeout=timeout)

    def _smoke_test_service(self, argv: list[str], command_str: str, timeout: int) -> ExecResult:
        """Smoke test for long-running processes (servers, bots, watchers).

        Starts the process, waits a probe window, checks if still alive.
        "Still running" = success for a service. Then terminates it.
        """
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            return ExecResult(
                command=command_str,
                exit_code=-1,
                stderr=f"Command not found: {argv[0]}",
            )

        probe_window = min(SMOKE_PROBE_WINDOW, timeout)
        time.sleep(probe_window)

        poll = proc.poll()
        if poll is None:
            # Process is still running — that's success for a service
            self.log(f"  [green]Service alive after {probe_window}s — healthy[/green]")
            # Terminate gracefully
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            return ExecResult(
                command=command_str,
                exit_code=0,
                stdout=(stdout or "")[-3000:],
                stderr=(stderr or "")[-3000:],
            )
        else:
            # Process exited during probe window — failure for a service
            stdout, stderr = proc.communicate()
            self.log(f"  [red]Service exited during probe window (exit {poll})[/red]")
            return ExecResult(
                command=command_str,
                exit_code=poll,
                stdout=(stdout or "")[-3000:],
                stderr=(stderr or "")[-3000:],
            )
