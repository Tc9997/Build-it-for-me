"""Template-first mode: the productized, narrow build appliance.

Exactly two archetypes: python_cli, fastapi_service.
Typed contract, deterministic policy, pinned templates, ownership manifest,
verifier authority. Reliability-focused.

Pipeline:
  RESEARCH → CONTRACT → ENVIRONMENT → POLICY → TEMPLATE → PLAN →
  BUILD+REVIEW → INTEGRATE → WRITE → SETUP → TEST+DEBUG → VERIFY →
  OPTIMIZE → ACCEPT
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
from build_loop.agents.planner import PlannerAgent
from build_loop.agents.researcher import ResearcherAgent
from build_loop.agents.reviewer import ReviewerAgent
from build_loop.agents.spec_compiler import SpecCompilerAgent
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
from build_loop.contract import BuildContract, CapabilityType
from build_loop.environment import EnvironmentSnapshot, capture_snapshot
from build_loop.plan_validation import validate_plan_coverage
from build_loop.policy import AutonomyMode, PolicyDecision, evaluate_policy
from build_loop.safety import safe_output_path
from build_loop.schemas import (
    BuildState,
    ContractState,
    EnvironmentState,
    PolicyState,
)
from build_loop.templates import registry as template_registry
from build_loop.templates.cache import CacheError, ensure_cached
from build_loop.templates.materialize import MaterializationError, materialize as materialize_template
from build_loop.templates.models import OwnershipManifest
from build_loop.templates.ownership import OwnershipViolationError, check_write_allowed
from build_loop.templates.registry import RegistryError
from build_loop.verifier import Verifier

console = Console()


class TemplateFirstOrchestrator:
    """Orchestrates the template-first build pipeline.

    Rejects anything outside python_cli and fastapi_service.
    All phase gates are hard. Verifier is the authority.
    """

    def __init__(self, output_dir: str | None = None, confirm_callback=None, run_optimizer: bool = False):
        self.output_dir = os.path.abspath(output_dir or os.path.join(os.getcwd(), "output"))
        self.state = BuildState(output_dir=self.output_dir)
        self._confirm = confirm_callback
        self._run_optimizer = run_optimizer

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

        # State
        self.contract: BuildContract | None = None
        self.env_snapshot: EnvironmentSnapshot | None = None
        self.policy_decision: PolicyDecision | None = None
        self._ownership_manifest: OwnershipManifest | None = None

    def resume(self, from_phase: str) -> str:
        """Resume from a saved phase. Loads state from disk.

        Reconstructs transient fields from persisted state:
        - contract from state.contract
        - _verification_result from state.verification
        - _ownership_manifest from .ownership.json on disk
        """
        state_path = Path(self.output_dir) / ".build_state" / "state.json"
        if not state_path.exists():
            raise PipelineError(f"No saved state at {state_path}")

        from build_loop.llm import reset_cost_tracking
        reset_cost_tracking()

        # Load and migrate saved state (fix malformed signals from older runs)
        import json as _json
        raw_state = _json.loads(state_path.read_text())
        _migrate_state(raw_state)
        self.state = BuildState.model_validate(raw_state)
        console.print(f"[bold]Resuming from phase: {from_phase}[/bold]")

        # Reconstruct transient fields from persisted state
        if self.state.contract:
            self.contract = self.state.contract.data

        if self.state.verification:
            from build_loop.verifier import VerificationResult
            self._verification_result = VerificationResult(**self.state.verification)

        # Reload ownership manifest from disk if it exists
        from build_loop.templates.materialize import MANIFEST_FILENAME
        manifest_path = Path(self.output_dir) / MANIFEST_FILENAME
        if manifest_path.exists():
            import json as _json
            self._ownership_manifest = OwnershipManifest.model_validate_json(
                manifest_path.read_text()
            )

        try:
            if from_phase == "write":
                phase("9", "WRITE", "Re-writing project to disk...")
                write_project(self.state, self.output_dir, self._safe_write)
                save_state(self.state, self.output_dir)

                from build_loop.analysis.post_write import run_post_write_checks
                archetype = self.contract.archetype if self.contract else ""
                pw_result = run_post_write_checks(
                    self.output_dir, archetype,
                    export_metadata=self.state.module_exports or None,
                )
                for check in pw_result.checks:
                    console.print(f"  [green]{check}[/green]")
                for err in pw_result.errors:
                    console.print(f"  [bold red]{err}[/bold red]")
                if not pw_result.passed:
                    raise PipelineError(f"Post-write checks failed: {pw_result.errors}")
                from_phase = "setup"

            if from_phase == "setup":
                phase("10", "SETUP", "Installing dependencies...")
                setup_environment(self.state, self.executor, self._venv_cmd)
                save_state(self.state, self.output_dir)

                from build_loop.analysis.post_write import run_post_setup_checks
                ps_result = run_post_setup_checks(self.output_dir)
                for check in ps_result.checks:
                    console.print(f"  [green]{check}[/green]")
                for err in ps_result.errors:
                    console.print(f"  [bold red]{err}[/bold red]")
                from_phase = "test"

            if from_phase == "test":
                phase("11", "TEST & DEBUG", "Running tests and fixing failures...")
                test_and_debug_loop(
                    self.state, self.executor, self.debugger,
                    self._venv_cmd, self._safe_write, self._read_files,
                )
                save_state(self.state, self.output_dir)
                from_phase = "verify"

            if from_phase == "verify":
                phase("12", "VERIFY", "Independent verification...")
                self._verify()
                save_state(self.state, self.output_dir)
                from_phase = "accept"

            if from_phase == "accept":
                phase("14", "ACCEPTANCE", "Final acceptance...")
                self._acceptance_check()
                save_state(self.state, self.output_dir)

        except (ModuleRejectedError, IntegrationFailedError, PipelineError, OwnershipViolationError) as e:
            console.print(f"\n[bold red]PIPELINE STOPPED: {e}[/bold red]")
            save_state(self.state, self.output_dir)
        except Exception as e:
            console.print(f"\n[bold red]PIPELINE CRASHED: {type(e).__name__}: {e}[/bold red]")
            save_state(self.state, self.output_dir)

        print_final_report(self.state)
        return self.output_dir

    def run(self, idea: str) -> str:
        """Run the full template-first pipeline. Returns the output directory."""
        from build_loop.llm import reset_cost_tracking
        reset_cost_tracking()
        self.state.idea = idea
        console.print(Panel(
            f"[bold]MODE: template_first[/bold]\n{idea}",
            title="[bold]PROJECT IDEA[/bold]",
        ))

        try:
            # Phase 1: Research (light mode for template_first — LLM knowledge only)
            phase("1", "RESEARCH", "Light research (no web search)...")
            self.state.research = self.researcher.run(idea, light=True)
            save_state(self.state, self.output_dir)

            # Phase 2: Contract — compact research summary, not full dump
            phase("2", "CONTRACT", "Compiling build contract...")
            research_summary = _compact_research(self.state.research)
            self.contract = self.spec_compiler.run(idea, research_summary)

            # Merge deterministic base signals with LLM-generated project-specific ones
            from build_loop.analysis.signal_derivation import derive_signals, merge_signals
            derived = derive_signals(self.contract)
            llm_signals = list(self.contract.success_signals)
            self.contract.success_signals = merge_signals(derived, llm_signals)
            log("contract", f"  {len(derived)} base + {len(llm_signals)} project signals")

            self.state.contract = ContractState(data=self.contract)
            save_state(self.state, self.output_dir)

            # Gate: reject unsupported archetypes
            if self.contract.archetype == "unsupported":
                raise PipelineError(
                    "This project does not fit a supported archetype "
                    "(python_cli or fastapi_service). Use --mode freeform instead."
                )

            # Phase 3: Environment
            phase("3", "ENVIRONMENT", "Capturing host capabilities...")
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
            save_state(self.state, self.output_dir)

            # Phase 4: Policy
            phase("4", "POLICY", "Evaluating build feasibility...")
            self.policy_decision = evaluate_policy(self.contract, self.env_snapshot)
            self.state.policy = PolicyState(data=self.policy_decision)
            save_state(self.state, self.output_dir)

            if self.policy_decision.autonomy_mode == AutonomyMode.REFUSE:
                raise PipelineError(f"Policy refused: {self.policy_decision.reasons}")

            # Phase 5: Template — resolve, cache, materialize
            phase("5", "TEMPLATE", "Resolving and materializing project template...")
            self._resolve_and_materialize_template()
            save_state(self.state, self.output_dir)

            # Phase 6: Plan
            phase("6", "PLAN", "Decomposing into modules and interfaces...")
            self._plan(research_summary)
            save_state(self.state, self.output_dir)

            self._checkpoint_gate("plan")

            # Phase 7: Build + Review
            phase("7", "BUILD", "Building modules with review loop...")
            build_all(self.state, self.builder, self.reviewer)
            save_state(self.state, self.output_dir)

            # Phase 8: Integrate
            phase("8", "INTEGRATE", "Wiring modules together...")
            template_files = sorted(self._ownership_manifest.files.keys()) if self._ownership_manifest else None
            self.state.integration = self.integrator.run(
                self.state.plan, self.state.artifacts,
                template_files=template_files,
                export_metadata=self.state.module_exports or None,
            )
            save_state(self.state, self.output_dir)
            if not self.state.integration.success:
                raise IntegrationFailedError(
                    f"Integration failed: {self.state.integration.issues}"
                )

            # Phase 9: Write (ownership-enforced)
            phase("9", "WRITE", "Writing project to disk...")
            write_project(self.state, self.output_dir, self._safe_write)
            save_state(self.state, self.output_dir)

            # Post-write checks (deterministic, no LLM)
            from build_loop.analysis.post_write import run_post_write_checks
            archetype = self.contract.archetype if self.contract else ""
            pw_result = run_post_write_checks(
                    self.output_dir, archetype,
                    export_metadata=self.state.module_exports or None,
                )
            for check in pw_result.checks:
                console.print(f"  [green]{check}[/green]")
            for err in pw_result.errors:
                console.print(f"  [bold red]{err}[/bold red]")
            if not pw_result.passed:
                raise PipelineError(f"Post-write checks failed: {pw_result.errors}")

            self._checkpoint_gate("setup")

            # Phase 10-13: Setup, Test, Verify, Optimize (skippable in DEGRADE)
            if not self._should_skip("setup"):
                phase("10", "SETUP", "Installing dependencies...")
                setup_environment(self.state, self.executor, self._venv_cmd)
                save_state(self.state, self.output_dir)

                # Post-setup checks (package importable — requires venv)
                from build_loop.analysis.post_write import run_post_setup_checks
                ps_result = run_post_setup_checks(self.output_dir)
                for check in ps_result.checks:
                    console.print(f"  [green]{check}[/green]")
                for err in ps_result.errors:
                    console.print(f"  [bold red]{err}[/bold red]")
                if not ps_result.passed:
                    console.print("  [yellow]Post-setup checks failed — package may not be importable[/yellow]")
            else:
                phase("10", "SETUP", "[SKIPPED — degraded mode]")

            if not self._should_skip("test"):
                phase("11", "TEST & DEBUG", "Running tests and fixing failures...")
                test_and_debug_loop(
                    self.state, self.executor, self.debugger,
                    self._venv_cmd, self._safe_write, self._read_files,
                )
                save_state(self.state, self.output_dir)
            else:
                phase("11", "TEST & DEBUG", "[SKIPPED — degraded mode]")

            if not self._should_skip("verify"):
                phase("12", "VERIFY", "Independent verification...")
                self._verify()
                save_state(self.state, self.output_dir)
            else:
                phase("12", "VERIFY", "[SKIPPED — degraded mode]")

            # Phase 13: Optimize — skipped by default (opt-in via optimize=True)
            if self._run_optimizer and not self._should_skip("optimize"):
                phase("13", "OPTIMIZE", "Optimizing...")
                optimize(
                    self.state, self.executor, self.optimizer, self.debugger,
                    self._venv_cmd, self._safe_write, self._read_files,
                )
                save_state(self.state, self.output_dir)
            else:
                phase("13", "OPTIMIZE", "[SKIPPED]")

            self._checkpoint_gate("acceptance")

            # Phase 14: Acceptance
            phase("14", "ACCEPTANCE", "Final acceptance...")
            self._acceptance_check()
            save_state(self.state, self.output_dir)

        except (ModuleRejectedError, IntegrationFailedError, PipelineError, OwnershipViolationError) as e:
            console.print(f"\n[bold red]PIPELINE STOPPED: {e}[/bold red]")
            save_state(self.state, self.output_dir)
        except Exception as e:
            console.print(f"\n[bold red]PIPELINE CRASHED: {type(e).__name__}: {e}[/bold red]")
            save_state(self.state, self.output_dir)

        print_final_report(self.state)
        return self.output_dir

    # ------------------------------------------------------------------
    # Template
    # ------------------------------------------------------------------

    def _resolve_and_materialize_template(self) -> None:
        """Resolve, cache, and materialize the template.

        All errors (RegistryError, CacheError, MaterializationError) are
        converted to PipelineError so they flow through the normal
        controlled error path with state preservation.

        The registry already verified the fixture against pinned hashes
        at initialization. The cache verifies content hash on populate.
        No redundant live verify_commit here.
        """
        if not self.contract:
            raise PipelineError("No contract")

        try:
            entry = template_registry.resolve(self.contract.archetype)
            cached = ensure_cached(entry)
            contract_hash = self.contract.canonical_hash()
            self._ownership_manifest = materialize_template(
                cached_template=cached,
                output_dir=Path(self.output_dir),
                project_name=self.contract.project_name,
                summary=self.contract.summary,
                template_id=entry.template_id,
                pinned_commit=entry.pinned_commit,
                contract_hash=contract_hash,
            )
            log("template", f"{len(self._ownership_manifest.files)} files materialized")
        except RegistryError as e:
            raise PipelineError(f"Template resolution failed: {e}")
        except CacheError as e:
            raise PipelineError(f"Template cache failed: {e}")
        except MaterializationError as e:
            raise PipelineError(f"Template materialization failed: {e}")

    # ------------------------------------------------------------------
    # Plan (contract-driven, template-aware)
    # ------------------------------------------------------------------

    def _plan(self, research_summary: str) -> None:
        contract_hash = self.contract.canonical_hash()
        template_files = sorted(self._ownership_manifest.files.keys()) if self._ownership_manifest else []

        plan_context = (
            f"BUILD CONTRACT (contract_hash={contract_hash}):\n"
            f"{json.dumps(self.contract.model_dump(), indent=2)}\n\n"
            f"TEMPLATE FILES ALREADY IN PROJECT (do not recreate these):\n"
            f"{json.dumps(template_files, indent=2)}\n\n"
            f"RESEARCH SUMMARY:\n{research_summary}"
        )
        self.state.plan = self.planner.run(plan_context)
        if self.contract:
            self.state.plan.run_mode = self.contract.run_mode
            self.state.plan.archetype = self.contract.archetype
            if not self.state.plan.contract_hash:
                self.state.plan.contract_hash = contract_hash

        validation = validate_plan_coverage(self.state.plan, self.contract)
        for w in validation.warnings:
            console.print(f"  [dim]{w}[/dim]")
        if not validation.valid:
            for e in validation.errors:
                console.print(f"  [bold red]{e}[/bold red]")
            raise PipelineError(f"Plan does not cover contract: {validation.errors}")

        print_plan(self.state.plan)

    # ------------------------------------------------------------------
    # Verify + Acceptance
    # ------------------------------------------------------------------

    def _verify(self) -> None:
        if not self.contract:
            return
        run_cmd = None
        if self.state.plan and self.state.plan.run_command and self.contract.run_mode == "service":
            run_cmd = self._venv_cmd(self.state.plan.run_command)
        verification = self.verifier.run(self.contract, run_command=run_cmd)
        self.state.verification = verification.model_dump()
        self._verification_result = verification

    def _acceptance_check(self) -> None:
        verification = getattr(self, "_verification_result", None)
        smoke_result = None
        if self.state.plan and self.state.plan.run_command:
            run_cmd = self._venv_cmd(self.state.plan.run_command)
            run_mode = getattr(self.state.plan, "run_mode", "batch")
            smoke_result = self.executor.smoke_test(run_cmd, run_mode=run_mode)
            self.state.exec_history.append(smoke_result)

        # Only send key files to acceptance — not the whole repo
        key_files = _select_acceptance_files(self._read_files())

        self.state.acceptance = self.acceptance.run(
            idea=self.state.idea,
            plan=self.state.plan,
            project_files=key_files,
            verification=verification,
            smoke_result=smoke_result,
        )

    # ------------------------------------------------------------------
    # Policy enforcement
    # ------------------------------------------------------------------

    def _checkpoint_gate(self, phase_name: str) -> None:
        if not self.policy_decision:
            return
        if self.policy_decision.autonomy_mode != AutonomyMode.CHECKPOINT:
            return
        if phase_name not in self.policy_decision.require_confirmation:
            return

        console.print(f"\n  [bold yellow]CHECKPOINT: confirmation required before '{phase_name}'[/bold yellow]")
        if self._confirm and self._confirm(phase_name, self.policy_decision.reasons):
            return
        raise PipelineError(
            f"Checkpoint at '{phase_name}' — no confirmation. "
            f"Reasons: {self.policy_decision.reasons}"
        )

    def _should_skip(self, phase_name: str) -> bool:
        if not self.policy_decision:
            return False
        if self.policy_decision.autonomy_mode != AutonomyMode.DEGRADE:
            return False
        return phase_name in set(self.policy_decision.skip_phases)

    # ------------------------------------------------------------------
    # File operations (ownership-enforced)
    # ------------------------------------------------------------------

    def _safe_write(self, relative_path: str, content: str) -> None:
        resolved = safe_output_path(self.output_dir, relative_path)
        if self._ownership_manifest:
            from build_loop.templates.models import FileOwner
            owner = self._ownership_manifest.owner_of(relative_path)

            if owner == FileOwner.TEMPLATE_LOCKED and resolved.exists():
                # Template-locked file already on disk — only skip if identical
                existing = resolved.read_text()
                if existing == content:
                    log("write", f"  skipped {relative_path} (template-locked, identical)")
                    return
                # Content differs — raise, this is a real conflict
                raise OwnershipViolationError(
                    f"Cannot overwrite template-locked path '{relative_path}' "
                    f"with different content. Template version takes precedence."
                )

            # USER_OWNED always raises (never silently skipped)
            check_write_allowed(self._ownership_manifest, relative_path)

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)

    def _venv_cmd(self, cmd: str) -> str:
        return venv_cmd(self.output_dir, cmd)

    def _read_files(self) -> dict[str, str]:
        return read_project_files(self.output_dir)


# ---------------------------------------------------------------------------
# State migration
# ---------------------------------------------------------------------------

def _migrate_state(raw: dict) -> None:
    """Migrate saved state from older schema versions.

    Fixes malformed success signals where command contains spaces
    and args is empty (shell-style commands from older spec compiler).
    """
    import shlex
    contract = raw.get("contract", {})
    if not contract:
        return
    data = contract.get("data", {})
    if not data:
        return
    for sig in data.get("success_signals", []):
        cmd = sig.get("command", "")
        args = sig.get("args", [])
        if " " in cmd:
            # Split shell-style command, prepend extra tokens to args
            parts = shlex.split(cmd)
            sig["command"] = parts[0]
            sig["args"] = parts[1:] + args


# ---------------------------------------------------------------------------
# Compact helpers to reduce token cost
# ---------------------------------------------------------------------------

def _compact_research(research) -> str:
    """Compact research report for downstream phases.

    Instead of dumping the full research JSON (can be 100k+ chars from
    web search), extract only the actionable parts.
    """
    parts = [f"Stack: {', '.join(research.recommended_stack)}"]
    for f in research.findings:
        parts.append(f"- {f.topic}: {f.summary[:200]}")
        if f.libraries:
            parts.append(f"  Libraries: {', '.join(f.libraries)}")
    if research.open_questions:
        parts.append(f"Open questions: {research.open_questions}")
    return "\n".join(parts)


def _select_acceptance_files(all_files: dict[str, str]) -> dict[str, str]:
    """Select only key files for acceptance review. Not the whole repo.

    Includes: __init__.py, entry points, config files, README, and
    any file under 2k chars. Skips large implementation files and tests.
    Total budget: 100k chars.
    """
    KEY_PATTERNS = (
        "__init__.py", "__main__.py", "main.py", "cli.py", "app.py",
        "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
        "README.md", "README", ".gitignore", "Dockerfile",
    )
    result = {}
    total = 0
    budget = 100_000

    # Key files first (always included if they fit)
    for path, content in sorted(all_files.items()):
        basename = path.rsplit("/", 1)[-1] if "/" in path else path
        if basename in KEY_PATTERNS:
            if total + len(content) < budget:
                result[path] = content
                total += len(content)

    # Small files next (< 2k, likely config or glue)
    for path, content in sorted(all_files.items()):
        if path in result:
            continue
        if len(content) < 2000 and total + len(content) < budget:
            result[path] = content
            total += len(content)

    return result
