"""Eval runner: executes tasks in both modes and captures structured results.

Each task gets its own isolated output directory. The runner captures
wall time, pipeline completion, verification results, and per-signal
pass/fail. No LLM judgment in the eval — only deterministic checks.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from rich.console import Console

from build_loop.eval.models import EvalRunResult, EvalSuiteResult, EvalTask
from build_loop.modes import BuildMode

console = Console()


def run_task(task: EvalTask, mode: BuildMode, output_base: Path) -> EvalRunResult:
    """Run a single eval task in the specified mode.

    Returns a structured result regardless of success or failure.
    """
    task_dir = output_base / f"{task.id}_{mode.value}"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    result = EvalRunResult(
        task_id=task.id,
        task_name=task.name,
        archetype=task.archetype,
        mode=mode.value,
        passed=False,
    )

    console.print(f"\n[bold]Running {task.id} ({task.name}) in {mode.value}...[/bold]")

    start = time.monotonic()
    try:
        from build_loop.agents.architect import ArchitectAgent
        agent = ArchitectAgent(output_dir=str(task_dir), mode=mode)
        agent.run(task.idea)

        result.pipeline_completed = True

        # Extract results from state
        state = agent.state
        result.debug_rounds = state.debug_rounds

        if state.acceptance:
            result.acceptance_verdict = state.acceptance.verdict.value if hasattr(state.acceptance.verdict, 'value') else str(state.acceptance.verdict)

        if state.verification:
            # Verification is stored as dict
            v = state.verification
            result.verification_passed = v.get("tier1_passed", False) and v.get("tier2_passed", False)
            for sr in v.get("tier1_results", []) + v.get("tier2_results", []):
                result.signal_results.append(sr)

        # Determine pass: verification passed (if it ran) + acceptance not failed
        if result.verification_passed is True:
            result.passed = True
        elif result.verification_passed is None:
            # Verification didn't run (degraded?) — check acceptance
            result.passed = result.acceptance_verdict == "pass"

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        console.print(f"  [red]Error: {result.error}[/red]")

    result.wall_time_seconds = round(time.monotonic() - start, 2)
    status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
    console.print(f"  {status} ({result.wall_time_seconds}s, {result.debug_rounds} debug rounds)")

    return result


def run_suite(
    tasks: list[EvalTask],
    mode: BuildMode,
    output_base: Path,
) -> EvalSuiteResult:
    """Run a suite of eval tasks and aggregate results."""
    suite = EvalSuiteResult(mode=mode.value)
    suite.total_tasks = len(tasks)

    for task in tasks:
        result = run_task(task, mode, output_base)
        suite.results.append(result)

        if result.error:
            suite.tasks_errored += 1
        elif result.passed:
            suite.tasks_passed += 1
        else:
            suite.tasks_failed += 1

        suite.total_wall_time += result.wall_time_seconds

    if suite.total_tasks > 0:
        suite.pass_rate = round(suite.tasks_passed / suite.total_tasks, 4)
        debug_rounds = [r.debug_rounds for r in suite.results]
        suite.avg_debug_rounds = round(sum(debug_rounds) / len(debug_rounds), 2)

    return suite
