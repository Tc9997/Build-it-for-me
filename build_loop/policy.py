"""Policy engine: deterministic decision tree over (contract, environment).

No LLM calls. Classifies requests by what's feasible on this machine right now,
then sets autonomy mode and constraints. The output is a PolicyDecision that the
architect reads before proceeding.

The rules are readable and auditable — a user can look at this file and
understand exactly why the system chose 'checkpoint' or 'refuse'.

Schema version is explicit for journal compatibility.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from build_loop.contract import BuildContract
from build_loop.environment import EnvironmentSnapshot

SCHEMA_VERSION = "1"


class AutonomyMode(str, Enum):
    """How much the pipeline can do without human confirmation."""
    PROCEED = "proceed"       # Fully autonomous — run everything
    CHECKPOINT = "checkpoint"  # Pause before execution phases for confirmation
    DEGRADE = "degrade"       # Skip phases that require missing capabilities
    REFUSE = "refuse"         # Cannot build this — explain why


class PolicyDecision(BaseModel):
    """The output of the policy engine. Read by the architect."""
    schema_version: str = SCHEMA_VERSION
    autonomy_mode: AutonomyMode
    reasons: list[str] = Field(
        default_factory=list,
        description="Why this mode was chosen — one reason per rule that fired"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking concerns"
    )
    blocked_capabilities: list[str] = Field(
        default_factory=list,
        description="Things the contract needs but the environment lacks"
    )
    skip_phases: list[str] = Field(
        default_factory=list,
        description="Phases to skip in degrade mode"
    )
    require_confirmation: list[str] = Field(
        default_factory=list,
        description="Phases that need human confirmation in checkpoint mode"
    )


def evaluate_policy(
    contract: BuildContract,
    env: EnvironmentSnapshot,
) -> PolicyDecision:
    """Deterministic policy evaluation. No LLM, no network, no side effects.

    Rules are evaluated in order. Each rule can escalate the autonomy mode
    but never de-escalate it (refuse > degrade > checkpoint > proceed).
    """
    mode = AutonomyMode.PROCEED
    reasons: list[str] = []
    warnings: list[str] = []
    blocked: list[str] = []
    skip: list[str] = []
    confirm: list[str] = []

    # ----- Rule: output directory must be writable -----
    if not env.output_dir_writable:
        mode = _escalate(mode, AutonomyMode.REFUSE)
        reasons.append("Output directory is not writable")

    # ----- Rule: Python must be available -----
    if not env.python_version:
        mode = _escalate(mode, AutonomyMode.REFUSE)
        reasons.append("Python is not available on this machine")

    # ----- Rule: missing required secrets -----
    missing_secrets = set(contract.secrets_required) - set(env.secrets_present)
    if missing_secrets:
        mode = _escalate(mode, AutonomyMode.CHECKPOINT)
        blocked.extend(f"secret:{s}" for s in missing_secrets)
        reasons.append(
            f"Missing required secrets: {sorted(missing_secrets)}. "
            "Set them in environment or .env before execution."
        )
        confirm.append("setup")

    # ----- Rule: external dependencies need Docker or network -----
    if contract.external_dependencies:
        needs_docker = any(
            kw in dep.lower()
            for dep in contract.external_dependencies
            for kw in ["docker", "container", "redis", "postgres", "mongo", "qdrant"]
        )
        if needs_docker and not env.docker_available:
            mode = _escalate(mode, AutonomyMode.DEGRADE)
            blocked.append("docker")
            reasons.append(
                "Contract requires containerized services but Docker is not available. "
                "External service setup will be skipped."
            )
            skip.append("external_services")

        needs_network = any(
            kw in dep.lower()
            for dep in contract.external_dependencies
            for kw in ["api", "http", "webhook", "oauth", "endpoint"]
        )
        if needs_network and not env.network_available:
            mode = _escalate(mode, AutonomyMode.DEGRADE)
            blocked.append("network")
            reasons.append(
                "Contract requires network access but network is unavailable. "
                "API-dependent features will be degraded."
            )
            skip.append("network_dependent_tests")

    # ----- Rule: service mode needs process management -----
    if contract.run_mode == "service":
        warnings.append(
            "Service-mode project: smoke test will use liveness probing, "
            "not exit-code checking."
        )

    # ----- Rule: too many open questions -----
    if len(contract.open_questions) > 3:
        mode = _escalate(mode, AutonomyMode.CHECKPOINT)
        reasons.append(
            f"Contract has {len(contract.open_questions)} open questions. "
            "Consider resolving them before building."
        )
        confirm.append("plan")

    # ----- Rule: no goals defined (contract is effectively empty) -----
    if not contract.goals:
        mode = _escalate(mode, AutonomyMode.REFUSE)
        reasons.append("Contract has no goals — nothing to build")

    # ----- Rule: no acceptance criteria -----
    if not contract.acceptance_criteria:
        mode = _escalate(mode, AutonomyMode.CHECKPOINT)
        reasons.append(
            "No acceptance criteria defined. Build will proceed but "
            "acceptance testing will be best-effort."
        )
        confirm.append("acceptance")

    return PolicyDecision(
        autonomy_mode=mode,
        reasons=reasons,
        warnings=warnings,
        blocked_capabilities=blocked,
        skip_phases=skip,
        require_confirmation=confirm,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Strict ordering: REFUSE > DEGRADE > CHECKPOINT > PROCEED
_MODE_SEVERITY = {
    AutonomyMode.PROCEED: 0,
    AutonomyMode.CHECKPOINT: 1,
    AutonomyMode.DEGRADE: 2,
    AutonomyMode.REFUSE: 3,
}


def _escalate(current: AutonomyMode, proposed: AutonomyMode) -> AutonomyMode:
    """Return the more restrictive of two modes. Never de-escalates."""
    if _MODE_SEVERITY[proposed] > _MODE_SEVERITY[current]:
        return proposed
    return current
