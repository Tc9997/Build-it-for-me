"""Tests for plan validation against contract coverage."""

import pytest

from build_loop.contract import BuildContract
from build_loop.plan_validation import validate_plan_coverage
from build_loop.schemas import BuildPlan, ModuleSpec, TaskSize


def _make_contract(**overrides) -> BuildContract:
    defaults = dict(
        project_name="test", summary="test",
        goals=["goal A", "goal B"],
        non_goals=["non-goal X"],
        acceptance_criteria=["test"],
    )
    defaults.update(overrides)
    return BuildContract(**defaults)


def _make_plan(**overrides) -> BuildPlan:
    defaults = dict(
        project_name="test", description="test", tech_stack=["python"],
        modules=[
            ModuleSpec(id="mod_a", name="A", description="does goal A", size=TaskSize.SMALL),
            ModuleSpec(id="mod_b", name="B", description="does goal B", size=TaskSize.SMALL),
        ],
        build_order=[["mod_a", "mod_b"]],
        goals_covered={
            "goal A": ["mod_a"],
            "goal B": ["mod_b"],
        },
        non_goals_acknowledged=["non-goal X"],
    )
    defaults.update(overrides)
    return BuildPlan(**defaults)


class TestPlanCoverage:
    """Plan must cover all contract goals."""

    def test_fully_covered_plan_is_valid(self):
        contract = _make_contract()
        plan = _make_plan(contract_hash=contract.canonical_hash())
        result = validate_plan_coverage(plan, contract)
        assert result.valid
        assert result.errors == []

    def test_missing_goal_coverage_is_invalid(self):
        contract = _make_contract()
        plan = _make_plan(
            goals_covered={"goal A": ["mod_a"]},  # Missing goal B
            contract_hash=contract.canonical_hash(),
        )
        result = validate_plan_coverage(plan, contract)
        assert not result.valid
        assert any("goal B" in e for e in result.errors)

    def test_nonexistent_module_in_coverage_is_invalid(self):
        contract = _make_contract()
        plan = _make_plan(
            goals_covered={
                "goal A": ["mod_a"],
                "goal B": ["mod_nonexistent"],
            },
            contract_hash=contract.canonical_hash(),
        )
        result = validate_plan_coverage(plan, contract)
        assert not result.valid
        assert any("mod_nonexistent" in e for e in result.errors)

    def test_wrong_contract_hash_is_invalid(self):
        contract = _make_contract()
        plan = _make_plan(contract_hash="wrong_hash_value")
        result = validate_plan_coverage(plan, contract)
        assert not result.valid
        assert any("mismatch" in e for e in result.errors)

    def test_empty_contract_hash_skips_check(self):
        """Empty hash (planner didn't set it) doesn't fail — it's set by architect."""
        contract = _make_contract()
        plan = _make_plan(contract_hash="")
        result = validate_plan_coverage(plan, contract)
        # No hash mismatch error — only check when hash is present
        assert not any("mismatch" in e for e in result.errors)

    def test_unacknowledged_non_goals_warn(self):
        contract = _make_contract()
        plan = _make_plan(
            non_goals_acknowledged=[],  # Didn't acknowledge non-goal X
            contract_hash=contract.canonical_hash(),
        )
        result = validate_plan_coverage(plan, contract)
        # Warnings, not errors
        assert result.valid
        assert any("non-goal" in w.lower() for w in result.warnings)

    def test_unmapped_modules_warn(self):
        """Modules not mapped to any goal produce a warning."""
        contract = _make_contract(goals=["goal A"])
        plan = _make_plan(
            goals_covered={"goal A": ["mod_a"]},
            # mod_b exists but isn't mapped to any goal
            contract_hash=contract.canonical_hash(),
        )
        result = validate_plan_coverage(plan, contract)
        assert result.valid  # Warning, not error
        assert any("mod_b" in w for w in result.warnings)


class TestContractHash:
    """Contract hashing must be canonical and stable."""

    def test_hash_is_deterministic(self):
        c1 = _make_contract()
        c2 = _make_contract()
        assert c1.canonical_hash() == c2.canonical_hash()

    def test_different_contracts_different_hashes(self):
        c1 = _make_contract(goals=["goal A"])
        c2 = _make_contract(goals=["goal B"])
        assert c1.canonical_hash() != c2.canonical_hash()

    def test_hash_is_sha256(self):
        h = _make_contract().canonical_hash()
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)


class TestBuildPlanVersioned:
    """BuildPlan schema_version must be Literal['1']."""

    def test_plan_has_schema_version(self):
        plan = _make_plan()
        assert plan.schema_version == "1"

    def test_wrong_version_fails_validation(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BuildPlan(
                schema_version="99",
                project_name="test", description="test",
            )
