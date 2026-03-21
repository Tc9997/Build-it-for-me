"""Architect agent: the orchestrator.

Takes an idea and autonomously delivers a working project through:
  RESEARCH → PLAN → BUILD → REVIEW → INTEGRATE → WRITE → SETUP → TEST+DEBUG → OPTIMIZE → ACCEPT

Every phase gate is a hard control-flow decision. Model output is validated
before any side effect. Review rejection blocks integration. Integration
failure stops the pipeline.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from build_loop.agents.base import Agent
from build_loop.agents.researcher import ResearcherAgent
from build_loop.agents.planner import PlannerAgent
from build_loop.agents.builder import BuilderAgent
from build_loop.agents.reviewer import ReviewerAgent
from build_loop.agents.integrator import IntegratorAgent
from build_loop.agents.executor import ExecutorAgent
from build_loop.agents.debugger import DebuggerAgent
from build_loop.agents.optimizer import OptimizerAgent
from build_loop.agents.acceptance import AcceptanceAgent
from build_loop.safety import PathTraversalError, safe_output_path
from build_loop.schemas import (
    AcceptanceVerdict,
    BuildArtifact,
    BuildPlan,
    BuildState,
    DebugFix,
    ModuleSpec,
    ReviewResult,
    ReviewVerdict,
    TaskStatus,
)

console = Console()

MAX_REVIEW_REVISIONS = 3
MAX_DEBUG_ROUNDS = 5


class ModuleRejectedError(Exception):
    """Raised when a module exhausts review revisions without approval."""

    def __init__(self, module_id: str, final_review: ReviewResult):
        self.module_id = module_id
        self.final_review = final_review
        super().__init__(
            f"Module '{module_id}' rejected after {MAX_REVIEW_REVISIONS} revisions: "
            f"{final_review.issues}"
        )


class IntegrationFailedError(Exception):
    """Raised when integration reports blocking issues."""


class PipelineError(Exception):
    """Raised for any phase-gate failure that should stop the pipeline."""


class ArchitectAgent(Agent):
    name = "architect"
    system_prompt = ""  # Architect doesn't call the LLM directly

    def __init__(self, output_dir: str | None = None):
        self.output_dir = os.path.abspath(output_dir or os.path.join(os.getcwd(), "output"))
        self.state = BuildState(output_dir=self.output_dir)

        # Sub-agents
        self.researcher = ResearcherAgent()
        self.planner = PlannerAgent()
        self.builder = BuilderAgent()
        self.reviewer = ReviewerAgent()
        self.integrator = IntegratorAgent()
        self.executor = ExecutorAgent(self.output_dir)
        self.debugger = DebuggerAgent()
        self.optimizer = OptimizerAgent()
        self.acceptance = AcceptanceAgent()

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================
    def run(self, idea: str) -> str:
        """Run the full autonomous build loop. Returns the output directory."""
        self.state.idea = idea

        console.print(Panel(idea, title="[bold]PROJECT IDEA[/bold]"))

        try:
            # Phase 1: Research
            self._phase("1", "RESEARCH", "Investigating feasibility and approach...")
            self.state.research = self.researcher.run(idea)
            self._print_research()
            self._save_state()

            # Phase 2: Plan
            self._phase("2", "PLAN", "Decomposing into modules and interfaces...")
            research_context = (
                f"{idea}\n\nRESEARCH FINDINGS:\n"
                f"{json.dumps(self.state.research.model_dump(), indent=2)}"
            )
            self.state.plan = self.planner.run(research_context)
            self._print_plan()
            self._save_state()

            # Phase 3: Build + Review (hard gate)
            self._phase("3", "BUILD", "Building modules with review loop...")
            self._build_all()
            self._save_state()

            # Phase 4: Integrate (hard gate)
            self._phase("4", "INTEGRATE", "Wiring modules together...")
            self.state.integration = self.integrator.run(
                self.state.plan, self.state.artifacts
            )
            self._save_state()

            if not self.state.integration.success:
                raise IntegrationFailedError(
                    f"Integration failed with {len(self.state.integration.issues)} issues: "
                    f"{self.state.integration.issues}"
                )

            # Phase 5: Write to disk (path-safe)
            self._phase("5", "WRITE", "Writing project to disk...")
            self._write_project()
            self._save_state()

            # Phase 6: Setup environment
            self._phase("6", "SETUP", "Installing dependencies...")
            self._setup_environment()
            self._save_state()

            # Phase 7: Execute + Debug loop
            self._phase("7", "TEST & DEBUG", "Running tests and fixing failures...")
            self._test_and_debug_loop()
            self._save_state()

            # Phase 8: Optimize
            self._phase("8", "OPTIMIZE", "Optimizing working code for performance and robustness...")
            self._optimize()
            self._save_state()

            # Phase 9: Acceptance
            self._phase("9", "ACCEPTANCE", "Validating against original intent...")
            self._acceptance_check()
            self._save_state()

        except (ModuleRejectedError, IntegrationFailedError, PipelineError) as e:
            console.print(f"\n[bold red]PIPELINE STOPPED: {e}[/bold red]")
            self._save_state()

        # Final report
        self._print_final_report()
        return self.output_dir

    # ==================================================================
    # PHASE IMPLEMENTATIONS
    # ==================================================================

    def _build_all(self) -> None:
        """Build all modules batch by batch. Failed modules are excluded from integration."""
        plan = self.state.plan

        for batch_idx, batch in enumerate(plan.build_order):
            console.print(f"\n[bold green]  Batch {batch_idx + 1}/{len(plan.build_order)}:[/bold green] {batch}")

            modules = {m.id: m for m in plan.modules if m.id in batch}

            with ThreadPoolExecutor(max_workers=max(len(modules), 1)) as pool:
                futures = {
                    pool.submit(self._build_and_review, mod, plan): mod.id
                    for mod in modules.values()
                }
                for future in as_completed(futures):
                    mid = futures[future]
                    try:
                        artifact, final_review = future.result()
                        self.state.artifacts[mid] = artifact
                        modules[mid].status = TaskStatus.APPROVED
                    except ModuleRejectedError as e:
                        console.print(f"  [bold red]{mid} REJECTED: {e.final_review.issues}[/bold red]")
                        modules[mid].status = TaskStatus.FAILED
                    except Exception as e:
                        console.print(f"  [bold red]{mid} failed: {e}[/bold red]")
                        modules[mid].status = TaskStatus.FAILED

        # Check if any modules were approved
        approved = [m for m in plan.modules if m.status == TaskStatus.APPROVED]
        if not approved:
            raise PipelineError("No modules passed review — cannot proceed to integration")

        failed = [m for m in plan.modules if m.status == TaskStatus.FAILED]
        if failed:
            console.print(
                f"\n  [yellow]Warning: {len(failed)} module(s) rejected and excluded from integration: "
                f"{[m.id for m in failed]}[/yellow]"
            )

    def _build_and_review(self, module: ModuleSpec, plan: BuildPlan) -> tuple[BuildArtifact, ReviewResult]:
        """Build → review → revise loop. Returns (artifact, final_review).

        Raises ModuleRejectedError if the module fails review after all revisions.
        """
        module.status = TaskStatus.IN_PROGRESS
        artifact = self.builder.run(module, plan)
        last_review = None

        for attempt in range(MAX_REVIEW_REVISIONS):
            module.status = TaskStatus.IN_REVIEW
            review = self.reviewer.run(module, artifact, plan)
            self.state.reviews.setdefault(module.id, []).append(review)
            last_review = review

            if review.verdict == ReviewVerdict.APPROVE:
                return artifact, review

            self.log(f"{module.id}: revision {attempt + 1}/{MAX_REVIEW_REVISIONS}")
            module.status = TaskStatus.REVISION
            artifact = self.builder.run(module, plan, revision_feedback=review)

        # Exhausted revisions — hard failure
        raise ModuleRejectedError(module.id, last_review)

    def _write_project(self) -> None:
        """Write all generated files to the output directory with path safety."""
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files_written = 0

        # Interface files
        if self.state.plan:
            for iface in self.state.plan.interfaces:
                if iface.code:
                    self._safe_write(iface.file_path, iface.code)
                    files_written += 1

        # Module files + tests
        for artifact in self.state.artifacts.values():
            for path, content in artifact.files.items():
                self._safe_write(path, content)
                files_written += 1
            for path, content in artifact.tests.items():
                self._safe_write(path, content)
                files_written += 1

        # Integration wiring
        if self.state.integration and self.state.integration.wiring_files:
            for path, content in self.state.integration.wiring_files.items():
                self._safe_write(path, content)
                files_written += 1

        self.log(f"wrote {files_written} files to {out}")

    def _setup_environment(self) -> None:
        """Run project setup commands (pip install, etc.)."""
        plan = self.state.plan
        if not plan or not plan.setup_commands:
            default_setup = [
                "python3 -m venv .venv",
                ".venv/bin/pip install --upgrade pip",
            ]
            req_path = Path(self.output_dir) / "requirements.txt"
            if req_path.exists():
                default_setup.append(".venv/bin/pip install -r requirements.txt")

            results = self.executor.setup_project(default_setup)
            self.state.exec_history.extend(results)
            return

        results = self.executor.setup_project(plan.setup_commands)
        self.state.exec_history.extend(results)

    def _test_and_debug_loop(self) -> None:
        """Run tests → debug → fix → rerun, up to MAX_DEBUG_ROUNDS."""
        plan = self.state.plan
        test_cmd = self._venv_cmd(plan.test_command) if plan else "pytest -v"
        previous_fixes: list[DebugFix] = []

        for round_num in range(MAX_DEBUG_ROUNDS):
            self.state.debug_rounds = round_num + 1
            console.print(f"\n  [bold]Debug round {round_num + 1}/{MAX_DEBUG_ROUNDS}[/bold]")

            test_result = self.executor.run_tests(test_cmd)
            self.state.exec_history.append(test_result)

            if test_result.success:
                console.print("  [bold green]All tests pass![/bold green]")
                return

            project_files = self._read_project_files()
            fix = self.debugger.run(
                error=test_result,
                plan=plan,
                project_files=project_files,
                previous_fixes=previous_fixes if previous_fixes else None,
            )
            previous_fixes.append(fix)
            self._apply_fix(fix)

        console.print(f"  [yellow]Exhausted {MAX_DEBUG_ROUNDS} debug rounds[/yellow]")

    def _optimize(self) -> None:
        """Run the optimizer on working code, then re-verify tests still pass."""
        plan = self.state.plan
        project_files = self._read_project_files()

        test_results = [r for r in self.state.exec_history if r.success and ("test" in r.command.lower() or "pytest" in r.command.lower())]
        test_result = test_results[-1] if test_results else None

        result = self.optimizer.run(
            plan=plan,
            project_files=project_files,
            test_result=test_result,
        )

        file_changes = result.get("file_changes", {})
        if not file_changes:
            console.print("  [dim]No optimizations needed[/dim]")
            return

        self.state.optimization_count = len(result.get("optimizations", []))

        for path, content in file_changes.items():
            self._safe_write(path, content)
            self.log(f"  optimized {path}")

        for dep in result.get("new_dependencies", []):
            r = self.executor.run_command(self._venv_cmd(f"pip install {dep}"))
            self.state.exec_history.append(r)

        console.print("\n  [bold]Re-running tests after optimization...[/bold]")
        test_cmd = self._venv_cmd(plan.test_command) if plan else "pytest -v"
        verify = self.executor.run_tests(test_cmd)
        self.state.exec_history.append(verify)

        if verify.success:
            console.print("  [bold green]Tests still pass after optimization[/bold green]")
        else:
            console.print("  [yellow]Optimization broke tests — entering debug loop to fix...[/yellow]")
            previous_fixes: list[DebugFix] = []
            for attempt in range(3):
                current_files = self._read_project_files()
                fix = self.debugger.run(
                    error=verify,
                    plan=plan,
                    project_files=current_files,
                    previous_fixes=previous_fixes if previous_fixes else None,
                )
                previous_fixes.append(fix)
                self._apply_fix(fix)

                verify = self.executor.run_tests(test_cmd)
                self.state.exec_history.append(verify)
                if verify.success:
                    console.print("  [bold green]Fixed — tests pass again[/bold green]")
                    return

            console.print("  [yellow]Could not fix optimization breakage — results may be degraded[/yellow]")

    def _acceptance_check(self) -> None:
        """Run acceptance testing against the original idea."""
        plan = self.state.plan
        project_files = self._read_project_files()

        test_results = [r for r in self.state.exec_history if "test" in r.command.lower() or "pytest" in r.command.lower()]
        test_result = test_results[-1] if test_results else None

        smoke_result = None
        if plan and plan.run_command:
            run_cmd = self._venv_cmd(plan.run_command)
            run_mode = getattr(plan, "run_mode", "batch")
            smoke_result = self.executor.smoke_test(run_cmd, run_mode=run_mode)
            self.state.exec_history.append(smoke_result)

        self.state.acceptance = self.acceptance.run(
            idea=self.state.idea,
            plan=plan,
            project_files=project_files,
            test_result=test_result,
            smoke_result=smoke_result,
        )

    # ==================================================================
    # SAFE FILE OPERATIONS
    # ==================================================================

    def _safe_write(self, relative_path: str, content: str) -> None:
        """Write a file, validating the path stays within the project root.

        Raises PathTraversalError (which stops the pipeline) if the path escapes.
        """
        resolved = safe_output_path(self.output_dir, relative_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)

    # ==================================================================
    # HELPERS
    # ==================================================================

    def _venv_cmd(self, cmd: str) -> str:
        """Prefix a command to use the project's venv if it exists."""
        venv = Path(self.output_dir) / ".venv" / "bin"
        if venv.exists():
            for tool in ("python", "pip", "pytest"):
                if cmd.startswith(tool + " ") or cmd == tool:
                    return str(venv / tool) + cmd[len(tool):]
        return cmd

    def _read_project_files(self) -> dict[str, str]:
        """Read all project files into a dict (for passing to LLM agents)."""
        out = Path(self.output_dir)
        files = {}
        for p in out.rglob("*"):
            if p.is_file() and not any(
                skip in str(p) for skip in [".venv", "__pycache__", ".build_state", ".git", "node_modules"]
            ):
                try:
                    content = p.read_text(errors="replace")
                    if len(content) < 50000:
                        files[str(p.relative_to(out))] = content
                except Exception:
                    pass
        return files

    def _apply_fix(self, fix: DebugFix) -> None:
        """Apply a debugger fix to the project files on disk (path-safe)."""
        for path, content in fix.file_changes.items():
            self._safe_write(path, content)
            self.log(f"  patched {path}")

        if fix.new_dependencies:
            for dep in fix.new_dependencies:
                result = self.executor.run_command(self._venv_cmd(f"pip install {dep}"))
                self.state.exec_history.append(result)

    def _save_state(self) -> None:
        state_dir = Path(self.output_dir) / ".build_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "state.json").write_text(self.state.model_dump_json(indent=2))

    # ==================================================================
    # DISPLAY
    # ==================================================================

    def _phase(self, num: str, name: str, desc: str) -> None:
        console.print(Panel(desc, title=f"[bold blue]PHASE {num}: {name}[/bold blue]"))

    def _print_research(self) -> None:
        r = self.state.research
        if not r:
            return
        console.print(f"  [bold]Feasibility:[/bold] {r.feasibility[:200]}")
        console.print(f"  [bold]Stack:[/bold] {', '.join(r.recommended_stack)}")
        if r.external_services:
            console.print(f"  [bold]External:[/bold] {', '.join(r.external_services)}")
        if r.open_questions:
            console.print(f"  [yellow]Open questions:[/yellow] {r.open_questions}")

    def _print_plan(self) -> None:
        plan = self.state.plan
        if not plan:
            return
        table = Table(title=f"{plan.project_name} — Build Plan")
        table.add_column("Module", style="cyan")
        table.add_column("Size")
        table.add_column("Deps")
        table.add_column("Files", style="dim")
        for m in plan.modules:
            table.add_row(m.id, m.size.value, ", ".join(m.dependencies) or "—", str(len(m.file_paths)))
        console.print(table)
        console.print(f"  Build order: {plan.build_order}")

    def _print_final_report(self) -> None:
        acc = self.state.acceptance
        status = "PASS" if acc and acc.verdict == AcceptanceVerdict.PASS else "FAIL"
        color = "green" if status == "PASS" else "red"

        report = Table(title="Build Report")
        report.add_column("Metric", style="bold")
        report.add_column("Value")
        report.add_row("Idea", self.state.idea[:100])
        report.add_row("Modules built", str(len(self.state.artifacts)))
        report.add_row("Debug rounds", str(self.state.debug_rounds))
        report.add_row("Optimizations", str(self.state.optimization_count))
        report.add_row("Acceptance", f"[{color}]{status}[/{color}]")
        report.add_row("Output", self.output_dir)

        if acc:
            report.add_row("Passed", ", ".join(acc.criteria_passed) or "—")
            if acc.criteria_failed:
                report.add_row("Failed", ", ".join(acc.criteria_failed))

        console.print(report)
