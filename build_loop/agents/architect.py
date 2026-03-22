"""Architect agent: thin router that delegates to mode-specific orchestrators.

Uses the engine/routing abstractions but preserves all existing behavior.
Imports are still lazy — a broken template registry doesn't break freeform.
"""

from __future__ import annotations

from build_loop.engine import RouteDecision
from build_loop.modes import BuildMode
from build_loop.routing import route, is_success

# Re-export exceptions so existing test imports still work
from build_loop.common.pipeline import (  # noqa: F401
    IntegrationFailedError,
    ModuleRejectedError,
    PipelineError,
)


class ArchitectAgent:
    """Routes to the appropriate build engine via RouteDecision.

    template_first is the default for supported archetypes.
    freeform is explicit-only and labeled experimental.
    """

    def __init__(
        self,
        output_dir: str | None = None,
        mode: BuildMode = BuildMode.TEMPLATE_FIRST,
        confirm_callback=None,
        run_optimizer: bool = False,
        mode_explicit: bool = False,
    ):
        self.mode = mode
        self.output_dir = output_dir
        self._confirm = confirm_callback
        self.decision: RouteDecision = route(mode, explicit=mode_explicit)

        # Lazy import: only load the selected mode's module
        if mode == BuildMode.TEMPLATE_FIRST:
            from build_loop.modes.template_first import TemplateFirstOrchestrator
            self._orchestrator = TemplateFirstOrchestrator(
                output_dir=output_dir,
                confirm_callback=confirm_callback,
                run_optimizer=run_optimizer,
            )
        elif mode == BuildMode.FREEFORM:
            from build_loop.modes.freeform import FreeformOrchestrator
            self._orchestrator = FreeformOrchestrator(
                output_dir=output_dir,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.state = self._orchestrator.state

    @property
    def contract(self):
        return getattr(self._orchestrator, "contract", None)

    @property
    def env_snapshot(self):
        return getattr(self._orchestrator, "env_snapshot", None)

    @property
    def policy_decision(self):
        return getattr(self._orchestrator, "policy_decision", None)

    def run(self, idea: str) -> str:
        """Run the full build pipeline."""
        result = self._orchestrator.run(idea)
        self.state = self._orchestrator.state
        return result

    def resume(self, from_phase: str) -> str:
        """Resume from a specific phase using saved state."""
        if not hasattr(self._orchestrator, "resume"):
            raise PipelineError(f"Resume not supported in {self.mode.value} mode")
        result = self._orchestrator.resume(from_phase)
        self.state = self._orchestrator.state
        return result

    def is_success(self) -> bool:
        """Check if the run succeeded according to the route decision's promise level."""
        if self.state.acceptance is None:
            return False
        v = self.state.acceptance.verdict
        verdict = str(v.value if hasattr(v, "value") else v)
        return is_success(self.decision, verdict)

    def __getattr__(self, name):
        """Delegate attribute access to the orchestrator for backward compat."""
        return getattr(self._orchestrator, name)
