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
    HttpProbeSignal,
)
from build_loop.environment import EnvironmentSnapshot, ToolAvailability
from build_loop.policy import AutonomyMode, evaluate_policy


def _make_contract(**overrides) -> BuildContract:
    defaults = dict(
        project_name="test", summary="test",
        goals=["test"], acceptance_criteria=["test"],
        archetype="python_cli",
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


class TestSystemToolCapability:
    """SYSTEM_TOOL must check env.tools by name, not assume available."""

    def test_tool_present_proceeds(self):
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(type=CapabilityType.SYSTEM_TOOL, name="git", required=True),
        ])
        env = _make_env(tools=[
            ToolAvailability(name="git", available=True, version="2.40.0"),
        ])
        decision = evaluate_policy(contract, env)
        assert decision.autonomy_mode == AutonomyMode.PROCEED

    def test_tool_missing_degrades(self):
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(type=CapabilityType.SYSTEM_TOOL, name="ffmpeg", required=True),
        ])
        env = _make_env(tools=[
            ToolAvailability(name="git", available=True),
            ToolAvailability(name="curl", available=True),
        ])
        decision = evaluate_policy(contract, env)
        assert decision.autonomy_mode == AutonomyMode.DEGRADE
        assert "system_tool:ffmpeg" in decision.blocked_capabilities

    def test_tool_in_list_but_not_available_degrades(self):
        """Tool is in the snapshot but marked unavailable."""
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(type=CapabilityType.SYSTEM_TOOL, name="docker", required=True),
        ])
        env = _make_env(tools=[
            ToolAvailability(name="docker", available=False),
        ])
        decision = evaluate_policy(contract, env)
        assert decision.autonomy_mode == AutonomyMode.DEGRADE

    def test_tool_empty_tools_list_degrades(self):
        """No tools detected at all — tool requirement degrades."""
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(type=CapabilityType.SYSTEM_TOOL, name="wkhtmltopdf", required=True),
        ])
        env = _make_env(tools=[])
        decision = evaluate_policy(contract, env)
        assert decision.autonomy_mode == AutonomyMode.DEGRADE

    def test_optional_tool_missing_warns(self):
        contract = _make_contract(capability_requirements=[
            CapabilityRequirement(type=CapabilityType.SYSTEM_TOOL, name="ffmpeg", required=False),
        ])
        env = _make_env(tools=[])
        decision = evaluate_policy(contract, env)
        assert decision.autonomy_mode == AutonomyMode.PROCEED
        assert any("ffmpeg" in w for w in decision.warnings)


class TestVerifierImpliedDependencies:
    """Policy must gate on tools the verifier needs, not just declared capabilities."""

    def test_http_probe_without_curl_degrades(self):
        """http_probe signals imply curl. Missing curl should degrade + skip verify."""
        contract = _make_contract(
            success_signals=[
                HttpProbeSignal(description="health", path="/health"),
            ],
        )
        # No curl in tools
        env = _make_env(tools=[
            ToolAvailability(name="git", available=True),
        ])
        decision = evaluate_policy(contract, env)
        assert decision.autonomy_mode == AutonomyMode.DEGRADE
        assert "system_tool:curl" in decision.blocked_capabilities
        assert "verify" in decision.skip_phases

    def test_http_probe_with_curl_proceeds(self):
        """http_probe with curl available should not trigger degrade."""
        contract = _make_contract(
            success_signals=[
                HttpProbeSignal(description="health", path="/health"),
            ],
        )
        env = _make_env(tools=[
            ToolAvailability(name="curl", available=True, version="8.0"),
        ])
        decision = evaluate_policy(contract, env)
        assert decision.autonomy_mode == AutonomyMode.PROCEED

    def test_no_http_probe_no_curl_check(self):
        """Without http_probe signals, missing curl doesn't matter."""
        contract = _make_contract()  # No signals
        env = _make_env(tools=[])  # No tools
        decision = evaluate_policy(contract, env)
        assert decision.autonomy_mode == AutonomyMode.PROCEED
        assert "system_tool:curl" not in decision.blocked_capabilities
