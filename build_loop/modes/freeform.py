"""Freeform mode: experimental autonomous generalist loop.

EXPERIMENTAL — not the product promise. Broader scope, best-effort quality.
Useful for exploration, fallback research, and benchmarking.

Pipeline:
  RESEARCH → PLAN → BUILD+REVIEW → INTEGRATE → WRITE → SETUP →
  TEST+DEBUG → OPTIMIZE → ACCEPT

No contract, no templates, no ownership manifest, no verifier.
LLM-based acceptance is the final judge (weaker guarantee).

Structured journal: every failure records a machine-readable FreeformIssue
so downstream tooling can inspect failed runs and drive targeted retries.

Retry policy (v1): setup and test failures get one automatic phase-local
retry. All other failure types stop the pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from build_loop.agents.acceptance import AcceptanceAgent
from build_loop.agents.builder import BuilderAgent
from build_loop.agents.debugger import DebuggerAgent
from build_loop.agents.executor import ExecutorAgent
from build_loop.agents.integrator import IntegratorAgent
from build_loop.agents.optimizer import OptimizerAgent
from build_loop.agents.planner import FreeformPlannerAgent
from build_loop.agents.researcher import ResearcherAgent
from build_loop.agents.reviewer import ReviewerAgent
from build_loop.common.pipeline import (
    IntegrationFailedError,
    ModuleRejectedError,
    PipelineError,
    build_all,
    log,
    optimize,
    phase,
    print_final_report,
    print_plan,
    read_project_files,
    save_state,
    setup_environment,
    test_and_debug_loop,
    venv_cmd,
    write_project,
)
from build_loop.freeform_journal import (
    FreeformJournal,
    IssueKind,
    IssueSeverity,
    IssueSource,
)
from build_loop.freeform_retry import RetryAction, plan_retry
from build_loop.safety import safe_output_path
from build_loop.schemas import BuildState

console = Console()


class FreeformOrchestrator:
    """Orchestrates the freeform (experimental) build pipeline.

    EXPERIMENTAL: broader scope, LLM-based acceptance, no contract or
    template guarantees. Use for exploration and benchmarking.
    """

    def __init__(self, output_dir: str | None = None):
        self.output_dir = os.path.abspath(output_dir or os.path.join(os.getcwd(), "output"))
        self.state = BuildState(output_dir=self.output_dir)
        self.journal = FreeformJournal()
        self.state.freeform_journal = self.journal
        self._current_phase: IssueSource = IssueSource.RESEARCH
        self._exception_already_journaled: bool = False

        # Sub-agents
        self.researcher = ResearcherAgent()
        self.planner = FreeformPlannerAgent()
        self.builder = BuilderAgent()
        self.reviewer = ReviewerAgent()
        self.integrator = IntegratorAgent()
        self.executor = ExecutorAgent(self.output_dir)
        self.debugger = DebuggerAgent()
        self.optimizer = OptimizerAgent()
        self.acceptance = AcceptanceAgent()

    def run(self, idea: str) -> str:
        """Run the freeform pipeline. Returns the output directory."""
        from build_loop.llm import reset_cost_tracking
        reset_cost_tracking()

        # Reset run-scoped state so the orchestrator is safe to reuse
        self._exception_already_journaled = False
        self._current_phase = IssueSource.RESEARCH
        self.journal = FreeformJournal()
        self.state = BuildState(output_dir=self.output_dir)
        self.state.freeform_journal = self.journal

        self.state.idea = idea
        console.print(Panel(
            f"[bold yellow]MODE: freeform (experimental)[/bold yellow]\n{idea}",
            title="[bold]PROJECT IDEA[/bold]",
        ))

        try:
            # Phase 1: Research
            self._current_phase = IssueSource.RESEARCH
            phase("1", "RESEARCH", "Investigating feasibility and approach...")
            self.state.research = self.researcher.run(idea)
            self.journal.record_attempt("research", success=True, summary="Research complete")
            save_state(self.state, self.output_dir)

            # Phase 2: Plan (prose-driven, no contract)
            self._current_phase = IssueSource.PLAN
            phase("2", "PLAN", "Decomposing into modules and interfaces...")
            research_json = json.dumps(self.state.research.model_dump(), indent=2)
            plan_context = f"{idea}\n\nRESEARCH FINDINGS:\n{research_json}"
            self.state.plan = self.planner.run(plan_context)
            print_plan(self.state.plan)
            self.journal.record_attempt("plan", success=True, summary="Plan complete")
            save_state(self.state, self.output_dir)

            # Phase 3: Build + Review
            self._current_phase = IssueSource.BUILD
            phase("3", "BUILD", "Building modules with review loop...")
            build_all(self.state, self.builder, self.reviewer)
            self.journal.record_attempt("build", success=True, summary="Build complete")
            save_state(self.state, self.output_dir)

            # Phase 4: Integrate
            self._current_phase = IssueSource.INTEGRATE
            phase("4", "INTEGRATE", "Wiring modules together...")
            self.state.integration = self.integrator.run(
                self.state.plan, self.state.artifacts
            )
            save_state(self.state, self.output_dir)
            if not self.state.integration.success:
                issues_text = "; ".join(self.state.integration.issues) if self.state.integration.issues else "unknown"
                self.journal.record_issue(
                    IssueKind.INTEGRATION_FAILURE,
                    IssueSource.INTEGRATE,
                    IssueSeverity.BLOCKING,
                    f"Integration failed: {issues_text}",
                    detail=self.state.integration.notes,
                    retryable=False,
                )
                self._exception_already_journaled = True
                self.journal.record_attempt(
                    "integrate", success=False,
                    summary=f"Integration failed: {issues_text}",
                )
                raise IntegrationFailedError(
                    f"Integration failed: {self.state.integration.issues}"
                )
            self.journal.record_attempt("integrate", success=True, summary="Integration complete")

            # Phase 5: Write (path-safe, no ownership enforcement)
            self._current_phase = IssueSource.WRITE
            phase("5", "WRITE", "Writing project to disk...")
            write_project(self.state, self.output_dir, self._safe_write)
            self.journal.record_attempt("write", success=True, summary="Files written to disk")
            save_state(self.state, self.output_dir)

            # Phase 6: Setup (with retry)
            self._run_setup()

            # Phase 7: Test + Debug (with retry)
            self._run_test_debug()

            # Phase 8: Optimize
            self._current_phase = IssueSource.OPTIMIZE
            phase("8", "OPTIMIZE", "Optimizing...")
            optimize(
                self.state, self.executor, self.optimizer, self.debugger,
                self._venv_cmd, self._safe_write, self._read_files,
            )
            self.journal.record_attempt("optimize", success=True, summary="Optimization complete")
            save_state(self.state, self.output_dir)

            # Phase 9: Acceptance (LLM-based, no verifier)
            self._current_phase = IssueSource.ACCEPTANCE
            phase("9", "ACCEPTANCE", "LLM-based acceptance (experimental)...")
            smoke_result = None
            if self.state.plan and self.state.plan.run_command:
                smoke_result = self.executor.smoke_test(
                    self._venv_cmd(self.state.plan.run_command),
                )
                self.state.exec_history.append(smoke_result)

            self.state.acceptance = self.acceptance.run(
                idea=self.state.idea,
                plan=self.state.plan,
                project_files=self._read_files(),
                verification=None,
                smoke_result=smoke_result,
                require_verification=False,  # Freeform has no verifier — LLM verdict stands
            )

            # Record acceptance outcome
            verdict_str = str(
                self.state.acceptance.verdict.value
                if hasattr(self.state.acceptance.verdict, "value")
                else self.state.acceptance.verdict
            )
            if verdict_str == "fail":
                self.journal.record_issue(
                    IssueKind.ACCEPTANCE_FAILURE,
                    IssueSource.ACCEPTANCE,
                    IssueSeverity.BLOCKING,
                    "Acceptance verdict: fail",
                    detail=self.state.acceptance.notes,
                    verdict="fail",
                    retryable=False,
                )
                self.journal.record_attempt(
                    "acceptance", success=False,
                    summary="Acceptance failed",
                )
            elif verdict_str == "incomplete":
                self.journal.record_issue(
                    IssueKind.ACCEPTANCE_INCOMPLETE,
                    IssueSource.ACCEPTANCE,
                    IssueSeverity.DEGRADED,
                    "Acceptance verdict: incomplete",
                    detail=self.state.acceptance.notes,
                    verdict="incomplete",
                    retryable=False,
                )
                self.journal.record_attempt(
                    "acceptance", success=True,
                    summary="Acceptance incomplete (best-effort pass)",
                )
            else:
                self.journal.record_attempt(
                    "acceptance", success=True,
                    summary="Acceptance passed",
                )
            save_state(self.state, self.output_dir)

        except (ModuleRejectedError, IntegrationFailedError, PipelineError) as e:
            # Record if not already captured by a phase-specific handler
            # (integration, setup retry, or test retry handlers set the flag)
            if not self._exception_already_journaled:
                self.journal.record_issue(
                    IssueKind.PIPELINE_ERROR,
                    self._current_phase,
                    IssueSeverity.BLOCKING,
                    str(e),
                    retryable=False,
                )
            console.print(f"\n[bold red]PIPELINE STOPPED: {e}[/bold red]")
            save_state(self.state, self.output_dir)
        except Exception as e:
            self.journal.record_issue(
                IssueKind.UNEXPECTED_CRASH,
                self._current_phase,
                IssueSeverity.BLOCKING,
                f"{type(e).__name__}: {e}",
                retryable=False,
            )
            console.print(f"\n[bold red]PIPELINE CRASHED: {type(e).__name__}: {e}[/bold red]")
            save_state(self.state, self.output_dir)

        # Print journal summary if there are issues
        if self.journal.issues:
            console.print(f"\n[dim]{self.journal.summary_text()}[/dim]")

        print_final_report(self.state)
        return self.output_dir

    # ------------------------------------------------------------------
    # Phase runners with retry support
    # ------------------------------------------------------------------

    def _run_setup(self) -> None:
        """Run setup with one automatic retry on failure."""
        self._current_phase = IssueSource.SETUP
        phase("6", "SETUP", "Installing dependencies...")
        attempt_num = 1

        try:
            setup_environment(self.state, self.executor, self._venv_cmd)
            self.journal.record_attempt(
                "setup", success=True, attempt_number=attempt_num,
                summary="Environment setup complete",
            )
            save_state(self.state, self.output_dir)
            return
        except PipelineError as e:
            issue = self.journal.record_issue(
                IssueKind.SETUP_FAILURE,
                IssueSource.SETUP,
                IssueSeverity.BLOCKING,
                f"Setup failed: {e}",
                retryable=True,
            )
            self.journal.record_attempt(
                "setup", success=False, attempt_number=attempt_num,
                summary=str(e),
            )
            save_state(self.state, self.output_dir)

        # Consult retry planner
        decision = plan_retry(issue, self.journal)
        if decision.action == RetryAction.STOP:
            console.print(f"  [dim]Retry planner: {decision.rationale}[/dim]")
            self._exception_already_journaled = True
            raise PipelineError(f"Setup failed (no retries remaining): {issue.summary}")

        # Retry once
        attempt_num = 2
        console.print(f"  [bold yellow]Retrying setup (attempt {attempt_num})...[/bold yellow]")
        try:
            setup_environment(self.state, self.executor, self._venv_cmd)
            self.journal.record_attempt(
                "setup", success=True, attempt_number=attempt_num,
                summary="Setup succeeded on retry",
            )
            save_state(self.state, self.output_dir)
        except PipelineError as e:
            self.journal.record_issue(
                IssueKind.SETUP_FAILURE,
                IssueSource.SETUP,
                IssueSeverity.BLOCKING,
                f"Setup failed on retry: {e}",
                retryable=True,
            )
            self.journal.record_attempt(
                "setup", success=False, attempt_number=attempt_num,
                summary=str(e),
            )
            save_state(self.state, self.output_dir)
            self._exception_already_journaled = True
            raise

    def _run_test_debug(self) -> None:
        """Run test/debug with one automatic retry on failure."""
        self._current_phase = IssueSource.TEST
        phase("7", "TEST & DEBUG", "Running tests and fixing failures...")
        attempt_num = 1

        try:
            test_and_debug_loop(
                self.state, self.executor, self.debugger,
                self._venv_cmd, self._safe_write, self._read_files,
            )
            self.journal.record_attempt(
                "test", success=True, attempt_number=attempt_num,
                summary="Tests passed",
            )
            save_state(self.state, self.output_dir)
            return
        except PipelineError as e:
            last_exec = self.state.exec_history[-1] if self.state.exec_history else None
            issue = self.journal.record_issue(
                IssueKind.TEST_FAILURE,
                IssueSource.TEST,
                IssueSeverity.BLOCKING,
                str(e),
                detail=last_exec.stderr[:2000] if last_exec and last_exec.stderr else "",
                command=last_exec.command if last_exec else "",
                retryable=True,
            )
            self.journal.record_attempt(
                "test", success=False, attempt_number=attempt_num,
                summary=str(e),
                command=last_exec.command if last_exec else "",
                exit_code=last_exec.exit_code if last_exec else None,
            )
            save_state(self.state, self.output_dir)

        # Consult retry planner
        decision = plan_retry(issue, self.journal)
        if decision.action == RetryAction.STOP:
            console.print(f"  [dim]Retry planner: {decision.rationale}[/dim]")
            self._exception_already_journaled = True
            raise PipelineError(f"Tests failed (no retries remaining): {issue.summary}")

        # Retry once — reset debug_rounds so test_and_debug_loop gets a fresh budget
        attempt_num = 2
        console.print(f"  [bold yellow]Retrying test/debug (attempt {attempt_num})...[/bold yellow]")
        self.state.debug_rounds = 0
        try:
            test_and_debug_loop(
                self.state, self.executor, self.debugger,
                self._venv_cmd, self._safe_write, self._read_files,
            )
            self.journal.record_attempt(
                "test", success=True, attempt_number=attempt_num,
                summary="Tests passed on retry",
            )
            save_state(self.state, self.output_dir)
        except PipelineError as e:
            last_exec = self.state.exec_history[-1] if self.state.exec_history else None
            self.journal.record_issue(
                IssueKind.TEST_FAILURE,
                IssueSource.TEST,
                IssueSeverity.BLOCKING,
                f"Tests failed on retry: {e}",
                detail=last_exec.stderr[:2000] if last_exec and last_exec.stderr else "",
                command=last_exec.command if last_exec else "",
                retryable=True,
            )
            self.journal.record_attempt(
                "test", success=False, attempt_number=attempt_num,
                summary=str(e),
                command=last_exec.command if last_exec else "",
                exit_code=last_exec.exit_code if last_exec else None,
            )
            save_state(self.state, self.output_dir)
            self._exception_already_journaled = True
            raise

    # ------------------------------------------------------------------
    # File operations (path-safe, no ownership)
    # ------------------------------------------------------------------

    def _safe_write(self, relative_path: str, content: str) -> None:
        resolved = safe_output_path(self.output_dir, relative_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)

    def _venv_cmd(self, cmd: str) -> str:
        return venv_cmd(self.output_dir, cmd)

    def _read_files(self) -> dict[str, str]:
        return read_project_files(self.output_dir)
