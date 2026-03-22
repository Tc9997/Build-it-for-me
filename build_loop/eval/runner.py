"""Eval runner: executes tasks in both modes and scores against corpus signals.

Pass/fail is determined by the eval harness's own expected_signals, NOT by
the build system's internal verifier or acceptance agent. Both modes are
scored by the same benchmark assertions — no mode has a structural advantage.

The build system's self-reported results (verification, acceptance) are
recorded for analysis but do not determine the eval outcome.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from rich.console import Console

from build_loop.eval.eval_verify import eval_verify
from build_loop.eval.models import EvalRunResult, EvalSuiteResult, EvalTask
from build_loop.modes import BuildMode

console = Console()


def run_task(task: EvalTask, mode: BuildMode, output_base: Path) -> EvalRunResult:
    """Run a single eval task and score against the corpus's expected_signals.

    Pass/fail is determined by eval_verify, not by the build system.
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
    )

    console.print(f"\n[bold]Running {task.id} ({task.name}) in {mode.value}...[/bold]")

    start = time.monotonic()
    try:
        from build_loop.agents.architect import ArchitectAgent
        agent = ArchitectAgent(output_dir=str(task_dir), mode=mode)
        agent.run(task.idea)

        # Extract self-reported results for analysis (not for scoring)
        state = agent.state
        result.debug_rounds = state.debug_rounds

        if state.acceptance:
            result.acceptance_verdict = (
                state.acceptance.verdict.value
                if hasattr(state.acceptance.verdict, "value")
                else str(state.acceptance.verdict)
            )

        if state.verification:
            v = state.verification
            result.verification_passed = (
                v.get("tier1_passed", False) and v.get("tier2_passed", False)
            )

        # Check pipeline_completed by looking for terminal-phase evidence
        # (acceptance ran, or at least write phase produced files)
        result.pipeline_completed = (
            state.acceptance is not None
            or len(list(task_dir.rglob("*.py"))) > 0
        )

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        console.print(f"  [red]Error: {result.error}[/red]")

    result.wall_time_seconds = round(time.monotonic() - start, 2)

    # ---------------------------------------------------------------
    # SCORING: eval-owned verification against corpus expected_signals
    # This is the authority for pass/fail — same for both modes.
    # ---------------------------------------------------------------
    if task.expected_signals:
        eval_signals = eval_verify(task, task_dir)
        result.signal_results = eval_signals
        result.passed = all(s["passed"] for s in eval_signals)
    else:
        # No corpus signals — fall back to "did the pipeline complete"
        result.passed = result.pipeline_completed

    status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
    signal_summary = ""
    if result.signal_results:
        sp = sum(1 for s in result.signal_results if s["passed"])
        signal_summary = f", {sp}/{len(result.signal_results)} signals"
    console.print(
        f"  {status} ({result.wall_time_seconds}s, "
        f"{result.debug_rounds} debug rounds{signal_summary})"
    )

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
