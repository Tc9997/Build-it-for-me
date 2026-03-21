"""Architect agent: the orchestrator.

Takes an idea and autonomously delivers a working project through:
  RESEARCH → CONTRACT → ENV → POLICY → PLAN → BUILD → INTEGRATE → WRITE →
  SETUP → TEST+DEBUG → VERIFY → OPTIMIZE → ACCEPT

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
from build_loop.agents.spec_compiler import SpecCompilerAgent
from build_loop.agents.planner import PlannerAgent
from build_loop.agents.builder import BuilderAgent
from build_loop.agents.reviewer import ReviewerAgent
from build_loop.agents.integrator import IntegratorAgent
from build_loop.agents.executor import ExecutorAgent
from build_loop.agents.debugger import DebuggerAgent
from build_loop.agents.optimizer import OptimizerAgent
from build_loop.agents.acceptance import AcceptanceAgent
from build_loop.verifier import Verifier, VerificationResult
from build_loop.contract import BuildContract
from build_loop.environment import EnvironmentSnapshot, capture_snapshot
from build_loop.policy import AutonomyMode, PolicyDecision, evaluate_policy
from build_loop.plan_validation import validate_plan_coverage
from build_loop.safety import PathTraversalError, safe_output_path
from build_loop.schemas import (
    AcceptanceVerdict,
    BuildArtifact,
    BuildPlan,
    BuildState,
    ContractState,
    DebugFix,
    EnvironmentState,
    ModuleSpec,
    PolicyState,
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

    def __init__(self, output_dir: str | None = None, confirm_callback=None):
        self.output_dir = os.path.abspath(output_dir or os.path.join(os.getcwd(), "output"))
        self.state = BuildState(output_dir=self.output_dir)

        # Confirmation callback for CHECKPOINT mode.
        # Signature: confirm_callback(phase_name: str, reasons: list[str]) -> bool
        # Returns True to proceed, False to stop.
        # If None in CHECKPOINT mode, the pipeline stops (fail-closed).
        self._confirm = confirm_callback

        # Sub-agents
        self.researcher = ResearcherAgent()
        self.spec_compiler = SpecCompilerAgent()
        self.planner = PlannerAgent()
        self.builder = BuilderAgent()
        self.reviewer = ReviewerAgent()
        self.integrator = IntegratorAgent()
        self.executor = ExecutorAgent(self.output_dir)
        self.debugger = DebuggerAgent()
        self.verifier = Verifier(self.output_dir)
        self.optimizer = OptimizerAgent()
        self.acceptance = AcceptanceAgent()

        # Populated during run
        self.contract: BuildContract | None = None
        self.env_snapshot: EnvironmentSnapshot | None = None
        self.policy_decision: PolicyDecision | None = None

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

            # Phase 2: Contract — compile prose + research into structured spec
            self._phase("2", "CONTRACT", "Compiling build contract from idea + research...")
            research_json = json.dumps(self.state.research.model_dump(), indent=2)
            self.contract = self.spec_compiler.run(idea, research_json)
            self.state.contract = ContractState(data=self.contract)
            self._print_contract()
            self._save_state()

            # Phase 3: Environment snapshot — what's available on this machine
            self._phase("3", "ENVIRONMENT", "Capturing host capabilities...")
            # Extract SYSTEM_TOOL names so snapshot probes them
            from build_loop.contract import CapabilityType
            required_tools = [
                cap.name for cap in self.contract.capability_requirements
                if cap.type == CapabilityType.SYSTEM_TOOL
            ]
            self.env_snapshot = capture_snapshot(
                output_dir=self.output_dir,
                required_secrets=self.contract.secrets_required,
                required_tools=required_tools,
            )
            self.state.environment = EnvironmentState(data=self.env_snapshot)
            self._print_environment()
            self._save_state()

            # Phase 4: Policy — can we build this here?
            self._phase("4", "POLICY", "Evaluating build feasibility...")
            self.policy_decision = evaluate_policy(self.contract, self.env_snapshot)
            self.state.policy = PolicyState(data=self.policy_decision)
            self._print_policy()
            self._save_state()

            # Gate: refuse stops the pipeline
            if self.policy_decision.autonomy_mode == AutonomyMode.REFUSE:
                raise PipelineError(
                    f"Policy refused build: {self.policy_decision.reasons}"
                )

            # Phase 5: Plan — contract-driven, not prose-driven
            self._phase("5", "PLAN", "Decomposing into modules and interfaces...")
            contract_hash = self.contract.canonical_hash()
            plan_context = (
                f"BUILD CONTRACT (contract_hash={contract_hash}):\n"
                f"{json.dumps(self.contract.model_dump(), indent=2)}\n\n"
                f"RESEARCH FINDINGS:\n{research_json}"
            )
            self.state.plan = self.planner.run(plan_context)
            # Carry run_mode and contract_hash from contract to plan
            if self.contract:
                self.state.plan.run_mode = self.contract.run_mode
                if not self.state.plan.contract_hash:
                    self.state.plan.contract_hash = contract_hash

            # Validate plan covers the contract — hard gate
            validation = validate_plan_coverage(self.state.plan, self.contract)
            for warn in validation.warnings:
                console.print(f"  [dim]Plan warning: {warn}[/dim]")
            if not validation.valid:
                for err in validation.errors:
                    console.print(f"  [bold red]Plan error: {err}[/bold red]")
                raise PipelineError(
                    f"Plan does not cover contract: {validation.errors}"
                )

            self._print_plan()
            self._save_state()

            # CHECKPOINT gate: plan requires confirmation before build
            self._checkpoint_gate("plan")

            # Phase 6: Build + Review (hard gate)
            self._phase("6", "BUILD", "Building modules with review loop...")
            self._build_all()
            self._save_state()

            # Phase 7: Integrate (hard gate)
            self._phase("7", "INTEGRATE", "Wiring modules together...")
            self.state.integration = self.integrator.run(
                self.state.plan, self.state.artifacts
            )
            self._save_state()

            if not self.state.integration.success:
                raise IntegrationFailedError(
                    f"Integration failed with {len(self.state.integration.issues)} issues: "
                    f"{self.state.integration.issues}"
                )

            # Phase 8: Write to disk (path-safe)
            self._phase("8", "WRITE", "Writing project to disk...")
            self._write_project()
            self._save_state()

            # CHECKPOINT gate: setup/test are side-effect phases
            self._checkpoint_gate("setup")

            # Phase 9: Setup environment (skippable in DEGRADE)
            if not self._should_skip("setup"):
                self._phase("9", "SETUP", "Installing dependencies...")
                self._setup_environment()
                self._save_state()
            else:
                self._phase("9", "SETUP", "[SKIPPED by policy — degraded mode]")

            # Phase 10: Execute + Debug loop (skippable in DEGRADE)
            if not self._should_skip("test"):
                self._phase("10", "TEST & DEBUG", "Running tests and fixing failures...")
                self._test_and_debug_loop()
                self._save_state()
            else:
                self._phase("10", "TEST & DEBUG", "[SKIPPED by policy — degraded mode]")

            # Phase 11: Verify (contract-derived, independent of builder tests)
            if not self._should_skip("verify"):
                self._phase("11", "VERIFY", "Independent verification against contract signals...")
                self._verify()
                self._save_state()
            else:
                self._phase("11", "VERIFY", "[SKIPPED by policy — degraded mode]")

            # Phase 12: Optimize (skippable in DEGRADE)
            if not self._should_skip("optimize"):
                self._phase("12", "OPTIMIZE", "Optimizing working code for performance and robustness...")
                self._optimize()
                self._save_state()
            else:
                self._phase("12", "OPTIMIZE", "[SKIPPED by policy — degraded mode]")

            # CHECKPOINT gate: acceptance requires confirmation
            self._checkpoint_gate("acceptance")

            # Phase 13: Acceptance (consumes VerificationResult, not raw prose)
            self._phase("13", "ACCEPTANCE", "Final acceptance against contract + verification...")
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

    def _verify(self) -> None:
        """Run the independent verifier against contract signals.

        The verifier is the authority for pass/fail. It executes contract
        signals deterministically, not via LLM judgment.

        For service-mode projects, passes the run_command so the verifier
        can start the service before executing http_probe signals.
        """
        if not self.contract:
            return
        run_cmd = None
        if self.state.plan and self.state.plan.run_command and self.contract.run_mode == "service":
            run_cmd = self._venv_cmd(self.state.plan.run_command)
        verification = self.verifier.run(self.contract, run_command=run_cmd)
        self.state.verification = verification.model_dump()
        self._verification_result = verification

    def _acceptance_check(self) -> None:
        """Final acceptance: consumes VerificationResult + residual gaps.

        The verifier is authoritative for machine-checkable signals.
        Acceptance only summarizes residual non-machine-checkable gaps
        (uncovered behavioral expectations, invariants, acceptance_criteria
        not covered by signals). It does NOT overwrite verifier verdicts.
        """
        plan = self.state.plan
        verification = getattr(self, "_verification_result", None)

        # Smoke test for service-mode projects
        smoke_result = None
        if plan and plan.run_command:
            run_cmd = self._venv_cmd(plan.run_command)
            run_mode = getattr(plan, "run_mode", "batch")
            smoke_result = self.executor.smoke_test(run_cmd, run_mode=run_mode)
            self.state.exec_history.append(smoke_result)

        self.state.acceptance = self.acceptance.run(
            idea=self.state.idea,
            plan=plan,
            project_files=self._read_project_files(),
            verification=verification,
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

    # ==================================================================
    # POLICY ENFORCEMENT
    # ==================================================================

    def _checkpoint_gate(self, phase_name: str) -> None:
        """Enforce CHECKPOINT mode: stop before side-effect phases unless confirmed.

        In CHECKPOINT mode, if the phase is in require_confirmation:
          - If a confirm_callback is provided and returns True, proceed.
          - Otherwise, raise PipelineError (fail-closed).
        In PROCEED or DEGRADE mode, this is a no-op.
        """
        if not self.policy_decision:
            return
        if self.policy_decision.autonomy_mode != AutonomyMode.CHECKPOINT:
            return
        if phase_name not in self.policy_decision.require_confirmation:
            return

        console.print(
            f"\n  [bold yellow]CHECKPOINT:[/bold yellow] Policy requires confirmation "
            f"before '{phase_name}'"
        )
        for r in self.policy_decision.reasons:
            console.print(f"    - {r}")

        if self._confirm:
            if self._confirm(phase_name, self.policy_decision.reasons):
                console.print(f"  [green]Confirmed — proceeding with {phase_name}[/green]")
                return
            else:
                raise PipelineError(
                    f"Checkpoint rejected at '{phase_name}': user declined to proceed"
                )
        else:
            # No callback — fail closed
            raise PipelineError(
                f"Checkpoint at '{phase_name}' with no confirmation callback — "
                f"cannot proceed in CHECKPOINT mode. "
                f"Reasons: {self.policy_decision.reasons}"
            )

    def _should_skip(self, phase_name: str) -> bool:
        """Check if a phase should be skipped in DEGRADE mode."""
        if not self.policy_decision:
            return False
        if self.policy_decision.autonomy_mode != AutonomyMode.DEGRADE:
            return False
        # Check both direct phase names and capability-derived skips
        skippable = set(self.policy_decision.skip_phases)
        return phase_name in skippable

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

    def _print_contract(self) -> None:
        c = self.contract
        if not c:
            return
        console.print(f"  [bold]Project:[/bold] {c.project_name}")
        console.print(f"  [bold]Goals:[/bold]")
        for g in c.goals:
            console.print(f"    - {g}")
        if c.non_goals:
            console.print(f"  [bold]Non-goals:[/bold]")
            for ng in c.non_goals:
                console.print(f"    - [dim]{ng}[/dim]")
        console.print(f"  [bold]Mode:[/bold] {c.run_mode}")
        console.print(f"  [bold]Signals:[/bold] {len(c.success_signals)} checkable")
        console.print(f"  [bold]Invariants:[/bold] {len(c.invariants)}")
        if c.open_questions:
            console.print(f"  [yellow]Open questions:[/yellow]")
            for q in c.open_questions:
                console.print(f"    ? {q}")

    def _print_environment(self) -> None:
        e = self.env_snapshot
        if not e:
            return
        console.print(f"  [bold]OS:[/bold] {e.os_name} {e.os_version} ({e.arch})")
        console.print(f"  [bold]Python:[/bold] {e.python_version}")
        available = [t.name for t in e.tools if t.available]
        missing = [t.name for t in e.tools if not t.available]
        if available:
            console.print(f"  [bold]Tools:[/bold] {', '.join(available)}")
        if missing:
            console.print(f"  [dim]Missing:[/dim] {', '.join(missing)}")
        console.print(f"  [bold]Docker:[/bold] {'yes' if e.docker_available else 'no'}")
        console.print(f"  [bold]Network:[/bold] {'yes' if e.network_available else 'no'}")
        if e.secrets_missing:
            console.print(f"  [yellow]Missing secrets:[/yellow] {e.secrets_missing}")

    def _print_policy(self) -> None:
        p = self.policy_decision
        if not p:
            return
        color = {
            AutonomyMode.PROCEED: "green",
            AutonomyMode.CHECKPOINT: "yellow",
            AutonomyMode.DEGRADE: "yellow",
            AutonomyMode.REFUSE: "red",
        }.get(p.autonomy_mode, "white")
        console.print(f"  [bold]Mode:[/bold] [{color}]{p.autonomy_mode.value}[/{color}]")
        for r in p.reasons:
            console.print(f"    - {r}")
        for w in p.warnings:
            console.print(f"    [dim]warning: {w}[/dim]")

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
        if acc and acc.verdict == AcceptanceVerdict.PASS:
            status, color = "PASS", "green"
        elif acc and acc.verdict.value == "incomplete":
            status, color = "INCOMPLETE", "yellow"
        else:
            status, color = "FAIL", "red"

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
