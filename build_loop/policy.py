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

from build_loop.contract import BuildContract, CapabilityType, HttpProbeSignal, SchemaValidSignal
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

    # ----- Rule: capability requirements vs environment -----
    for cap in contract.capability_requirements:
        available = _check_capability(cap.type, env, cap.name)
        if not available and cap.required:
            mode = _escalate(mode, AutonomyMode.DEGRADE)
            blocked.append(f"{cap.type.value}:{cap.name}")
            phases_affected = cap.affects_phases or ["setup", "test", "optimize"]
            skip.extend(phases_affected)
            reasons.append(
                f"Contract requires {cap.type.value} capability '{cap.name}' "
                f"but it is not available. Phases {phases_affected} will be skipped."
            )
        elif not available and not cap.required:
            warnings.append(
                f"Optional capability '{cap.name}' ({cap.type.value}) is not available. "
                "Some features may be degraded."
            )

    # ----- Rule: verifier-implied tool dependencies -----
    # Success signals imply tools the verifier needs at runtime.
    # If those tools are missing, verification will fail — gate early.
    _signal_tool_deps: dict[type, str] = {
        HttpProbeSignal: "curl",
    }
    for signal in contract.success_signals:
        tool_name = _signal_tool_deps.get(type(signal))
        if tool_name:
            tool_available = _check_capability(CapabilityType.SYSTEM_TOOL, env, tool_name)
            if not tool_available:
                mode = _escalate(mode, AutonomyMode.DEGRADE)
                blocked.append(f"system_tool:{tool_name}")
                skip.append("verify")
                reasons.append(
                    f"Contract has {type(signal).__name__} signals but "
                    f"'{tool_name}' is not available. Verify phase will be skipped."
                )
                break  # One missing tool is enough to skip verify

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
        skip_phases=sorted(set(skip)),  # Deduplicate
        require_confirmation=sorted(set(confirm)),  # Deduplicate
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


def _check_capability(cap_type: CapabilityType, env: EnvironmentSnapshot, cap_name: str = "") -> bool:
    """Check if the environment satisfies a capability requirement.

    Deterministic — maps capability types to environment fields.
    SYSTEM_TOOL matches against env.tools by name (case-insensitive).
    """
    if cap_type == CapabilityType.DOCKER:
        return env.docker_available
    elif cap_type == CapabilityType.NETWORK:
        return env.network_available
    elif cap_type == CapabilityType.SERVICE:
        # Services typically need Docker or network; check both
        return env.docker_available or env.network_available
    elif cap_type == CapabilityType.SYSTEM_TOOL:
        # Match against the tools detected in environment snapshot
        name_lower = cap_name.lower()
        return any(
            t.available and t.name.lower() == name_lower
            for t in env.tools
        )
    elif cap_type == CapabilityType.HARDWARE:
        # Can't detect hardware from snapshot — always missing
        return False
    return True


def _escalate(current: AutonomyMode, proposed: AutonomyMode) -> AutonomyMode:
    """Return the more restrictive of two modes. Never de-escalates."""
    if _MODE_SEVERITY[proposed] > _MODE_SEVERITY[current]:
        return proposed
    return current
