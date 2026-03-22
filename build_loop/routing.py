"""Route decision logic: selects an engine for a build request.

Currently deterministic from BuildMode. Future: could consider the idea
text, contract archetype, or environment capabilities to auto-route.
"""

from __future__ import annotations

from build_loop.engine import EngineCapabilities, PromiseLevel, RouteDecision
from build_loop.modes import BuildMode


# Static capability declarations for each engine
TEMPLATE_FIRST_CAPABILITIES = EngineCapabilities(
    name="template_first",
    promise_level=PromiseLevel.VERIFIED,
    supports_resume=True,
    supports_verify=True,
    supported_archetypes=["python_cli", "fastapi_service"],
    description="Productized, narrow. Contract-driven, template-backed, verifier authority.",
)

FREEFORM_CAPABILITIES = EngineCapabilities(
    name="freeform",
    promise_level=PromiseLevel.BEST_EFFORT,
    supports_resume=False,
    supports_verify=False,
    supported_archetypes=[],  # Accepts anything
    description="Experimental, broad. Prose-driven, LLM-judged.",
)


def route(mode: BuildMode) -> RouteDecision:
    """Select an engine based on the requested build mode.

    Returns a RouteDecision capturing the selection, confidence, and rationale.
    """
    if mode == BuildMode.TEMPLATE_FIRST:
        return RouteDecision(
            engine_name="template_first",
            promise_level=PromiseLevel.VERIFIED,
            confidence=1.0,
            rationale="User selected template_first (default). Verifier-backed, narrow scope.",
            mode_value=mode.value,
        )
    elif mode == BuildMode.FREEFORM:
        return RouteDecision(
            engine_name="freeform",
            promise_level=PromiseLevel.BEST_EFFORT,
            confidence=1.0,
            rationale="User explicitly selected freeform (experimental).",
            mode_value=mode.value,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def is_success(decision: RouteDecision, verdict: str) -> bool:
    """Determine if a verdict counts as success for the given route decision.

    template_first (VERIFIED): only "pass" is success.
    freeform (BEST_EFFORT): "pass" or "incomplete" is success; only "fail" is failure.
    """
    if decision.promise_level == PromiseLevel.VERIFIED:
        return verdict == "pass"
    else:
        return verdict != "fail"
