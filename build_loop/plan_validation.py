"""Plan validation: checks that a BuildPlan covers the contract.

This is deterministic — no LLM. It checks:
  1. contract_hash matches the contract
  2. Every contract goal has at least one module mapped in goals_covered
  3. All non_goals are acknowledged
  4. All module IDs in goals_covered actually exist in the plan
"""

from __future__ import annotations

from dataclasses import dataclass, field

from build_loop.contract import BuildContract
from build_loop.schemas import BuildPlan


@dataclass
class PlanValidationResult:
    """Result of validating a BuildPlan against a BuildContract."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_plan_coverage(plan: BuildPlan, contract: BuildContract) -> PlanValidationResult:
    """Check that the plan adequately covers the contract.

    Returns a result with errors (blocking) and warnings (non-blocking).
    """
    result = PlanValidationResult()
    module_ids = {m.id for m in plan.modules}

    # 1. Contract hash must match
    expected_hash = contract.canonical_hash()
    if plan.contract_hash and plan.contract_hash != expected_hash:
        result.errors.append(
            f"Plan contract_hash mismatch: plan has {plan.contract_hash[:16]}... "
            f"but contract is {expected_hash[:16]}..."
        )
        result.valid = False

    # 2. Every contract goal must have at least one module mapped
    uncovered_goals = []
    for goal in contract.goals:
        mapped_modules = plan.goals_covered.get(goal, [])
        if not mapped_modules:
            uncovered_goals.append(goal)
        else:
            # Check that mapped module IDs actually exist
            for mid in mapped_modules:
                if mid not in module_ids:
                    result.errors.append(
                        f"Goal '{goal[:50]}' maps to module '{mid}' which does not exist in plan"
                    )
                    result.valid = False

    if uncovered_goals:
        result.errors.append(
            f"{len(uncovered_goals)} contract goal(s) have no module mapped: "
            f"{uncovered_goals}"
        )
        result.valid = False

    # 3. Non-goals should be acknowledged
    unacknowledged = [
        ng for ng in contract.non_goals
        if ng not in plan.non_goals_acknowledged
    ]
    if unacknowledged:
        result.warnings.append(
            f"{len(unacknowledged)} non-goal(s) not acknowledged by planner: "
            f"{unacknowledged}"
        )

    # 4. No modules should exist without being mapped to any goal
    mapped_modules = set()
    for mods in plan.goals_covered.values():
        mapped_modules.update(mods)
    unmapped = module_ids - mapped_modules
    if unmapped:
        result.warnings.append(
            f"{len(unmapped)} module(s) not mapped to any goal: {sorted(unmapped)}"
        )

    return result
