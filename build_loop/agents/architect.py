"""Architect agent: thin router that delegates to mode-specific orchestrators.

Two modes:
  template_first: Productized, narrow. python_cli and fastapi_service only.
  freeform: Experimental, broad. Any project type, best-effort.

Imports are lazy: template_first is only imported when that mode is selected.
A broken template registry does not prevent freeform from loading.
"""

from __future__ import annotations

from build_loop.modes import BuildMode

# Re-export exceptions so existing test imports still work
from build_loop.common.pipeline import (  # noqa: F401
    IntegrationFailedError,
    ModuleRejectedError,
    PipelineError,
)


class ArchitectAgent:
    """Routes to the appropriate build mode orchestrator.

    template_first is the default for supported archetypes.
    freeform is explicit-only and labeled experimental.

    Mode-specific modules are imported lazily so a template registry
    failure (e.g. missing fixtures, drifted pins) does not break
    freeform mode.
    """

    def __init__(
        self,
        output_dir: str | None = None,
        mode: BuildMode = BuildMode.TEMPLATE_FIRST,
        confirm_callback=None,
    ):
        self.mode = mode
        self.output_dir = output_dir
        self._confirm = confirm_callback

        # Lazy import: only load the selected mode's module
        if mode == BuildMode.TEMPLATE_FIRST:
            from build_loop.modes.template_first import TemplateFirstOrchestrator
            self._orchestrator = TemplateFirstOrchestrator(
                output_dir=output_dir,
                confirm_callback=confirm_callback,
            )
        elif mode == BuildMode.FREEFORM:
            from build_loop.modes.freeform import FreeformOrchestrator
            self._orchestrator = FreeformOrchestrator(
                output_dir=output_dir,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Expose state for test compatibility
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
        """Run the build pipeline in the configured mode."""
        result = self._orchestrator.run(idea)
        self.state = self._orchestrator.state
        return result

    def __getattr__(self, name):
        """Delegate attribute access to the orchestrator for backward compat."""
        return getattr(self._orchestrator, name)
