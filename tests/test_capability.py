"""Tests for structured capability requirements in policy.

Proves:
- Policy matches on CapabilityType enum, not substring search
- affects_phases from the requirement flows into skip_phases
- Optional vs required capabilities produce different modes
- No substring inference from free text
"""

import pytest

from build_loop.contract import (
    BuildContract,
    CapabilityRequirement,
    CapabilityType,
)
from build_loop.environment import EnvironmentSnapshot
from build_loop.policy import AutonomyMode, evaluate_policy


def _make_contract(**overrides) -> BuildContract:
    defaults = dict(
        project_name="test", summary="test",
        goals=["test"], acceptance_criteria=["test"],
    )
    defaults.update(overrides)
    return BuildContract(**defaults)


def _make_env(**overrides) -> EnvironmentSnapshot:
    defaults = dict(
        os_name="Darwin", arch="arm64",
        python_version="3.12.0", python_path="/usr/bin/python3",
        docker_available=True, network_available=True,
        output_dir_writable=True, tools=[],
    )
    defaults.update(overrides)
    return EnvironmentSnapshot(**defaults)


class TestCapabilityMatching:
    """Policy must match on typed capabilities, not free text."""

    def test_docker_required_and_missing_degrades(self):
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(
                type=CapabilityType.DOCKER, name="Redis",
                required=True, affects_phases=["setup", "test"],
            ),
        ])
        decision = evaluate_policy(contract, _make_env(docker_available=False))
        assert decision.autonomy_mode == AutonomyMode.DEGRADE
        assert "docker:Redis" in decision.blocked_capabilities
        assert "setup" in decision.skip_phases
        assert "test" in decision.skip_phases

    def test_docker_required_and_present_proceeds(self):
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(
                type=CapabilityType.DOCKER, name="Redis",
                required=True,
            ),
        ])
        decision = evaluate_policy(contract, _make_env(docker_available=True))
        assert decision.autonomy_mode == AutonomyMode.PROCEED

    def test_network_required_and_missing_degrades(self):
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(
                type=CapabilityType.NETWORK, name="GitHub API",
                required=True, affects_phases=["test", "optimize"],
            ),
        ])
        decision = evaluate_policy(contract, _make_env(network_available=False))
        assert decision.autonomy_mode == AutonomyMode.DEGRADE
        assert "network:GitHub API" in decision.blocked_capabilities

    def test_optional_capability_missing_warns_only(self):
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(
                type=CapabilityType.DOCKER, name="Grafana",
                required=False,
            ),
        ])
        decision = evaluate_policy(contract, _make_env(docker_available=False))
        assert decision.autonomy_mode == AutonomyMode.PROCEED
        assert any("Grafana" in w for w in decision.warnings)

    def test_hardware_always_missing(self):
        """Hardware capabilities cannot be detected from snapshot — always degrade."""
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(
                type=CapabilityType.HARDWARE, name="Bluetooth",
                required=True, affects_phases=["test"],
            ),
        ])
        decision = evaluate_policy(contract, _make_env())
        assert decision.autonomy_mode == AutonomyMode.DEGRADE
        assert "hardware:Bluetooth" in decision.blocked_capabilities

    def test_affects_phases_flows_to_skip_phases(self):
        """The specific phases from the requirement must appear in skip_phases."""
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(
                type=CapabilityType.DOCKER, name="Postgres",
                required=True, affects_phases=["setup", "verify"],
            ),
        ])
        decision = evaluate_policy(contract, _make_env(docker_available=False))
        assert "setup" in decision.skip_phases
        assert "verify" in decision.skip_phases
        # Phases not listed should NOT be skipped
        assert "optimize" not in decision.skip_phases

    def test_no_substring_matching(self):
        """Free-text strings should NOT trigger policy rules.
        Only typed CapabilityRequirements matter."""
        # This contract has goals mentioning Docker but no capability_requirements
        contract = _make_contract(
            goals=["Run Docker container for Redis", "Use Docker Compose"],
        )
        decision = evaluate_policy(contract, _make_env(docker_available=False))
        # Should NOT degrade — no typed requirement, just prose in goals
        assert decision.autonomy_mode == AutonomyMode.PROCEED

    def test_multiple_capabilities(self):
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(type=CapabilityType.DOCKER, name="Redis", required=True),
            CapabilityRequirement(type=CapabilityType.NETWORK, name="Stripe API", required=True),
        ])
        # Docker present, network missing
        decision = evaluate_policy(contract, _make_env(docker_available=True, network_available=False))
        assert decision.autonomy_mode == AutonomyMode.DEGRADE
        assert "network:Stripe API" in decision.blocked_capabilities
        assert "docker:Redis" not in decision.blocked_capabilities

    def test_service_with_docker_or_network(self):
        """SERVICE type passes if either Docker or network is available."""
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(type=CapabilityType.SERVICE, name="PostgreSQL", required=True),
        ])
        # No docker, but network available
        decision = evaluate_policy(contract, _make_env(docker_available=False, network_available=True))
        assert decision.autonomy_mode == AutonomyMode.PROCEED
