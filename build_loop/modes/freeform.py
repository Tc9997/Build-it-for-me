"""Freeform mode: experimental autonomous generalist loop.

EXPERIMENTAL — not the product promise. Broader scope, best-effort quality.
Useful for exploration, fallback research, and benchmarking.

Pipeline:
  RESEARCH → PLAN → BUILD+REVIEW → INTEGRATE → WRITE → SETUP →
  TEST+DEBUG → OPTIMIZE → ACCEPT

No contract, no templates, no ownership manifest, no verifier.
LLM-based acceptance is the final judge (weaker guarantee).
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
        self.state.idea = idea
        console.print(Panel(
            f"[bold yellow]MODE: freeform (experimental)[/bold yellow]\n{idea}",
            title="[bold]PROJECT IDEA[/bold]",
        ))

        try:
            # Phase 1: Research
            phase("1", "RESEARCH", "Investigating feasibility and approach...")
            self.state.research = self.researcher.run(idea)
            save_state(self.state, self.output_dir)

            # Phase 2: Plan (prose-driven, no contract)
            phase("2", "PLAN", "Decomposing into modules and interfaces...")
            research_json = json.dumps(self.state.research.model_dump(), indent=2)
            plan_context = f"{idea}\n\nRESEARCH FINDINGS:\n{research_json}"
            self.state.plan = self.planner.run(plan_context)
            print_plan(self.state.plan)
            save_state(self.state, self.output_dir)

            # Phase 3: Build + Review
            phase("3", "BUILD", "Building modules with review loop...")
            build_all(self.state, self.builder, self.reviewer)
            save_state(self.state, self.output_dir)

            # Phase 4: Integrate
            phase("4", "INTEGRATE", "Wiring modules together...")
            self.state.integration = self.integrator.run(
                self.state.plan, self.state.artifacts
            )
            save_state(self.state, self.output_dir)
            if not self.state.integration.success:
                raise IntegrationFailedError(
                    f"Integration failed: {self.state.integration.issues}"
                )

            # Phase 5: Write (path-safe, no ownership enforcement)
            phase("5", "WRITE", "Writing project to disk...")
            write_project(self.state, self.output_dir, self._safe_write)
            save_state(self.state, self.output_dir)

            # Phase 6: Setup
            phase("6", "SETUP", "Installing dependencies...")
            setup_environment(self.state, self.executor, self._venv_cmd)
            save_state(self.state, self.output_dir)

            # Phase 7: Test + Debug
            phase("7", "TEST & DEBUG", "Running tests and fixing failures...")
            test_and_debug_loop(
                self.state, self.executor, self.debugger,
                self._venv_cmd, self._safe_write, self._read_files,
            )
            save_state(self.state, self.output_dir)

            # Phase 8: Optimize
            phase("8", "OPTIMIZE", "Optimizing...")
            optimize(
                self.state, self.executor, self.optimizer, self.debugger,
                self._venv_cmd, self._safe_write, self._read_files,
            )
            save_state(self.state, self.output_dir)

            # Phase 9: Acceptance (LLM-based, no verifier)
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
                verification=None,  # No verifier in freeform mode
                smoke_result=smoke_result,
            )
            save_state(self.state, self.output_dir)

        except (ModuleRejectedError, IntegrationFailedError, PipelineError) as e:
            console.print(f"\n[bold red]PIPELINE STOPPED: {e}[/bold red]")
            save_state(self.state, self.output_dir)
        except Exception as e:
            console.print(f"\n[bold red]PIPELINE CRASHED: {type(e).__name__}: {e}[/bold red]")
            save_state(self.state, self.output_dir)

        print_final_report(self.state)
        return self.output_dir

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
