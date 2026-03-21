"""Tests for the policy engine — deterministic rules over (contract, environment)."""

import pytest

from build_loop.contract import BuildContract, CapabilityRequirement, CapabilityType
from build_loop.environment import EnvironmentSnapshot, ToolAvailability
from build_loop.policy import AutonomyMode, PolicyDecision, evaluate_policy


def _make_contract(**overrides) -> BuildContract:
    defaults = dict(
        project_name="test",
        summary="test",
        goals=["do something"],
        acceptance_criteria=["it works"],
        archetype="python_cli",
    )
    defaults.update(overrides)
    return BuildContract(**defaults)


def _make_env(**overrides) -> EnvironmentSnapshot:
    defaults = dict(
        os_name="Darwin",
        arch="arm64",
        python_version="3.12.0",
        python_path="/usr/bin/python3",
        docker_available=True,
        network_available=True,
        output_dir_writable=True,
        tools=[],
    )
    defaults.update(overrides)
    return EnvironmentSnapshot(**defaults)


class TestPolicyEngine:
    """Policy must be deterministic: same inputs → same outputs."""

    def test_happy_path_proceeds(self):
        """A simple contract on a capable machine should proceed."""
        decision = evaluate_policy(_make_contract(), _make_env())
        assert decision.autonomy_mode == AutonomyMode.PROCEED
        assert decision.reasons == []

    def test_no_goals_refuses(self):
        """A contract with no goals has nothing to build."""
        decision = evaluate_policy(
            _make_contract(goals=[]),
            _make_env(),
        )
        assert decision.autonomy_mode == AutonomyMode.REFUSE
        assert any("no goals" in r.lower() for r in decision.reasons)

    def test_missing_secrets_checkpoints(self):
        """Missing required secrets should trigger checkpoint, not refuse."""
        decision = evaluate_policy(
            _make_contract(secrets_required=["MY_API_KEY"]),
            _make_env(secrets_missing=["MY_API_KEY"]),
        )
        assert decision.autonomy_mode == AutonomyMode.CHECKPOINT
        assert "secret:MY_API_KEY" in decision.blocked_capabilities

    def test_docker_needed_but_missing_degrades(self):
        """Needing Docker without it should degrade, not refuse."""
        decision = evaluate_policy(
            _make_contract(capability_requirements=[
                CapabilityRequirement(type=CapabilityType.DOCKER, name="Redis", required=True),
            ]),
            _make_env(docker_available=False),
        )
        assert decision.autonomy_mode == AutonomyMode.DEGRADE
        assert "docker:Redis" in decision.blocked_capabilities

    def test_network_needed_but_missing_degrades(self):
        decision = evaluate_policy(
            _make_contract(capability_requirements=[
                CapabilityRequirement(type=CapabilityType.NETWORK, name="API", required=True),
            ]),
            _make_env(network_available=False),
        )
        assert decision.autonomy_mode == AutonomyMode.DEGRADE
        assert "network:API" in decision.blocked_capabilities

    def test_unwritable_output_refuses(self):
        decision = evaluate_policy(
            _make_contract(),
            _make_env(output_dir_writable=False),
        )
        assert decision.autonomy_mode == AutonomyMode.REFUSE

    def test_no_python_refuses(self):
        decision = evaluate_policy(
            _make_contract(),
            _make_env(python_version=""),
        )
        assert decision.autonomy_mode == AutonomyMode.REFUSE

    def test_many_open_questions_checkpoints(self):
        decision = evaluate_policy(
            _make_contract(open_questions=["q1", "q2", "q3", "q4"]),
            _make_env(),
        )
        assert decision.autonomy_mode == AutonomyMode.CHECKPOINT

    def test_no_acceptance_criteria_checkpoints(self):
        decision = evaluate_policy(
            _make_contract(acceptance_criteria=[]),
            _make_env(),
        )
        assert decision.autonomy_mode == AutonomyMode.CHECKPOINT

    def test_service_mode_warns(self):
        decision = evaluate_policy(
            _make_contract(run_mode="service"),
            _make_env(),
        )
        # Should proceed but with a warning about liveness probing
        assert decision.autonomy_mode == AutonomyMode.PROCEED
        assert any("service" in w.lower() for w in decision.warnings)

    def test_escalation_never_deescalates(self):
        """Multiple issues: the most restrictive mode wins."""
        decision = evaluate_policy(
            _make_contract(
                goals=[],  # → REFUSE
                secrets_required=["KEY"],  # → CHECKPOINT (weaker)
            ),
            _make_env(secrets_missing=["KEY"]),
        )
        assert decision.autonomy_mode == AutonomyMode.REFUSE

    def test_schema_version_present(self):
        decision = evaluate_policy(_make_contract(), _make_env())
        assert decision.schema_version == "1"

    def test_deterministic(self):
        """Same inputs must produce identical outputs."""
        c = _make_contract(secrets_required=["KEY"])
        e = _make_env(secrets_missing=["KEY"])
        d1 = evaluate_policy(c, e)
        d2 = evaluate_policy(c, e)
        assert d1.model_dump() == d2.model_dump()
