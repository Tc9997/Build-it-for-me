"""Executor agent: runs commands in the project directory and captures results."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console

from build_loop.schemas import ExecResult

console = Console()

DEFAULT_TIMEOUT = 120  # seconds


class ExecutorAgent:
    """Not an LLM agent — this one actually runs code."""

    name = "executor"

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        console.print(f"[bold magenta][{self.name}][/bold magenta] {msg}")

    def run_command(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> ExecResult:
        """Run a shell command in the project directory."""
        self.log(f"$ {command}")
        try:
            proc = subprocess.run(
                command,
                shell=True,
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
                # Show last few lines of stderr
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

    def smoke_test(self, run_command: str, timeout: int = 30) -> ExecResult:
        """Run the project briefly to see if it starts without crashing."""
        self.log(f"smoke test: {run_command}")
        return self.run_command(run_command, timeout=timeout)
