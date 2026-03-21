"""Architect agent: the orchestrator. Takes an idea, coordinates all other agents, delivers a project."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from build_loop.agents.base import Agent
from build_loop.agents.planner import PlannerAgent
from build_loop.agents.builder import BuilderAgent
from build_loop.agents.reviewer import ReviewerAgent
from build_loop.agents.integrator import IntegratorAgent
from build_loop.schemas import (
    BuildArtifact,
    BuildPlan,
    BuildState,
    ModuleSpec,
    ReviewVerdict,
    TaskStatus,
)

console = Console()

MAX_REVISIONS = 3


SYSTEM = """\
You are the Architect agent — the senior technical lead of an automated build system. \
You take a user's project idea and ask 2-3 short, critical clarifying questions before \
handing off to the Planner.

Focus your questions on:
- What is the primary tech stack / language?
- What is the deployment target (CLI, web app, library, etc.)?
- Any hard constraints (must use X, can't use Y)?

Respond with a JSON object:
{
  "questions": ["string", ...]
}

Only ask questions whose answers would materially change the build plan. If the idea is \
already specific enough, return an empty questions list.

Respond with ONLY the JSON object.
"""


class ArchitectAgent(Agent):
    name = "architect"
    system_prompt = SYSTEM

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output")
        self.state = BuildState(output_dir=self.output_dir)
        self.planner = PlannerAgent()
        self.builder = BuilderAgent()
        self.reviewer = ReviewerAgent()
        self.integrator = IntegratorAgent()

    # ------------------------------------------------------------------
    # Step 1: Clarify
    # ------------------------------------------------------------------
    def clarify(self, idea: str) -> list[str]:
        """Ask clarifying questions about the idea. Returns list of questions (may be empty)."""
        self.state.idea = idea
        data = self.call_json(f"PROJECT IDEA:\n{idea}")
        return data.get("questions", [])

    # ------------------------------------------------------------------
    # Step 2: Plan
    # ------------------------------------------------------------------
    def plan(self, clarifications: str = "") -> BuildPlan:
        """Generate the build plan."""
        if clarifications:
            self.state.clarifications.append({"user": clarifications})

        all_context = self.state.idea
        for c in self.state.clarifications:
            all_context += "\n" + json.dumps(c)

        self.state.plan = self.planner.run(all_context)
        self._save_state()
        return self.state.plan

    # ------------------------------------------------------------------
    # Step 3: Build (with review loop)
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Execute the build plan batch by batch."""
        plan = self.state.plan
        if not plan:
            raise RuntimeError("No plan — call plan() first")

        for batch_idx, batch in enumerate(plan.build_order):
            console.print(Panel(
                f"Batch {batch_idx + 1}/{len(plan.build_order)}: {batch}",
                title="[bold green]BUILD BATCH[/bold green]",
            ))

            modules = {m.id: m for m in plan.modules if m.id in batch}

            # Build modules in parallel within a batch
            with ThreadPoolExecutor(max_workers=len(modules)) as pool:
                futures = {
                    pool.submit(self._build_and_review, module, plan): module.id
                    for module in modules.values()
                }
                for future in as_completed(futures):
                    mid = futures[future]
                    try:
                        artifact = future.result()
                        self.state.artifacts[mid] = artifact
                        modules[mid].status = TaskStatus.APPROVED
                    except Exception as e:
                        console.print(f"[bold red][architect] module {mid} failed: {e}[/bold red]")
                        modules[mid].status = TaskStatus.FAILED

            self._save_state()

    def _build_and_review(self, module: ModuleSpec, plan: BuildPlan) -> BuildArtifact:
        """Build a module, review it, revise if needed. Returns approved artifact."""
        module.status = TaskStatus.IN_PROGRESS
        artifact = self.builder.run(module, plan)

        for attempt in range(MAX_REVISIONS):
            module.status = TaskStatus.IN_REVIEW
            review = self.reviewer.run(module, artifact, plan)

            # Track review history
            self.state.reviews.setdefault(module.id, []).append(review)

            if review.verdict == ReviewVerdict.APPROVE:
                return artifact

            # Revision needed
            self.log(f"{module.id}: revision {attempt + 1}/{MAX_REVISIONS}")
            module.status = TaskStatus.REVISION
            artifact = self.builder.run(module, plan, revision_feedback=review)

        # If we exhaust revisions, accept what we have with a warning
        self.log(f"{module.id}: accepting after {MAX_REVISIONS} revisions (not fully approved)")
        return artifact

    # ------------------------------------------------------------------
    # Step 4: Integrate
    # ------------------------------------------------------------------
    def integrate(self) -> None:
        """Wire all approved modules together."""
        plan = self.state.plan
        if not plan:
            raise RuntimeError("No plan")

        self.state.integration = self.integrator.run(plan, self.state.artifacts)
        self._save_state()

    # ------------------------------------------------------------------
    # Step 5: Write to disk
    # ------------------------------------------------------------------
    def write_project(self) -> str:
        """Write all files to the output directory. Returns the output path."""
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        files_written = 0

        # Write interface files
        if self.state.plan:
            for iface in self.state.plan.interfaces:
                if iface.code:
                    self._write_file(out / iface.file_path, iface.code)
                    files_written += 1

        # Write module files
        for artifact in self.state.artifacts.values():
            for path, content in artifact.files.items():
                self._write_file(out / path, content)
                files_written += 1
            for path, content in artifact.tests.items():
                self._write_file(out / path, content)
                files_written += 1

        # Write integration wiring
        if self.state.integration and self.state.integration.wiring_files:
            for path, content in self.state.integration.wiring_files.items():
                self._write_file(out / path, content)
                files_written += 1

        self.log(f"wrote {files_written} files to {out}")
        return str(out)

    def _write_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        state_dir = Path(self.output_dir) / ".build_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "state.json").write_text(self.state.model_dump_json(indent=2))

    # ------------------------------------------------------------------
    # Full run (convenience)
    # ------------------------------------------------------------------
    def run(self, idea: str, clarifications: str = "") -> str:
        """Run the full build loop end-to-end. Returns the output directory."""
        console.print(Panel(idea, title="[bold]PROJECT IDEA[/bold]"))

        # Clarify
        questions = self.clarify(idea)
        if questions and not clarifications:
            console.print("[bold yellow]Architect has questions:[/bold yellow]")
            for q in questions:
                console.print(f"  ? {q}")
            # In non-interactive mode, proceed without answers
            console.print("[dim]Proceeding without answers (pass clarifications to override)[/dim]")

        # Plan
        console.print(Panel("Planning...", title="[bold blue]PHASE 1: PLAN[/bold blue]"))
        self.plan(clarifications)

        # Build
        console.print(Panel("Building...", title="[bold blue]PHASE 2: BUILD[/bold blue]"))
        self.build()

        # Integrate
        console.print(Panel("Integrating...", title="[bold blue]PHASE 3: INTEGRATE[/bold blue]"))
        self.integrate()

        # Write
        console.print(Panel("Writing files...", title="[bold blue]PHASE 4: WRITE[/bold blue]"))
        output = self.write_project()

        console.print(Panel(f"[bold green]Done! Project written to: {output}[/bold green]"))
        return output
