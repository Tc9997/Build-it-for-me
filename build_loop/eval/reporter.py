"""Eval reporter: machine-readable JSON + human-readable summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from build_loop.eval.models import EvalSuiteResult

console = Console()


def save_results(results: list[EvalSuiteResult], output_path: Path) -> None:
    """Save eval results as JSON for CI/tracking."""
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suites": [r.model_dump() for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


def print_comparison(suites: list[EvalSuiteResult]) -> None:
    """Print a human-readable comparison of eval suite results."""
    console.print("\n")

    # Summary table
    table = Table(title="Eval Results")
    table.add_column("Mode", style="bold")
    table.add_column("Pass", style="green")
    table.add_column("Fail", style="red")
    table.add_column("Error", style="yellow")
    table.add_column("Rate")
    table.add_column("Time")
    table.add_column("Avg Debug")

    for s in suites:
        table.add_row(
            s.mode,
            str(s.tasks_passed),
            str(s.tasks_failed),
            str(s.tasks_errored),
            f"{s.pass_rate:.0%}",
            f"{s.total_wall_time:.0f}s",
            f"{s.avg_debug_rounds:.1f}",
        )

    console.print(table)

    # Per-task detail
    for s in suites:
        console.print(f"\n[bold]{s.mode}[/bold] — detail:")
        for r in s.results:
            status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            if r.error:
                status = "[yellow]ERROR[/yellow]"
            console.print(
                f"  {status} {r.task_id} ({r.task_name}) "
                f"— {r.wall_time_seconds}s, {r.debug_rounds} debug rounds"
                + (f" [{r.error[:60]}]" if r.error else "")
            )

    # Head-to-head comparison if we have both modes
    if len(suites) == 2:
        console.print("\n[bold]Head-to-head:[/bold]")
        s0, s1 = suites
        ids_0 = {r.task_id: r for r in s0.results}
        ids_1 = {r.task_id: r for r in s1.results}
        common = sorted(set(ids_0) & set(ids_1))

        wins = {s0.mode: 0, s1.mode: 0, "tie": 0}
        for tid in common:
            r0, r1 = ids_0[tid], ids_1[tid]
            # Errored runs are always non-pass — r.passed is guaranteed False
            if r0.passed and not r1.passed:
                wins[s0.mode] += 1
                suffix = " (opponent errored)" if r1.error else ""
                console.print(f"  {tid}: {s0.mode} wins{suffix}")
            elif r1.passed and not r0.passed:
                wins[s1.mode] += 1
                suffix = " (opponent errored)" if r0.error else ""
                console.print(f"  {tid}: {s1.mode} wins{suffix}")
            elif r0.passed and r1.passed:
                wins["tie"] += 1
                # Compare on wall time
                faster = s0.mode if r0.wall_time_seconds < r1.wall_time_seconds else s1.mode
                console.print(f"  {tid}: both pass ({faster} faster)")
            else:
                wins["tie"] += 1
                console.print(f"  {tid}: both fail")

        console.print(
            f"\n  {s0.mode}: {wins[s0.mode]} wins | "
            f"{s1.mode}: {wins[s1.mode]} wins | "
            f"ties: {wins['tie']}"
        )
