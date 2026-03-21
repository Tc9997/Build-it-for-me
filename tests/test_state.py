"""Tests for typed BuildState fields and schema versioning."""

import json

import pytest
from pydantic import ValidationError

from build_loop.contract import BuildContract
from build_loop.environment import EnvironmentSnapshot
from build_loop.policy import AutonomyMode, PolicyDecision
from build_loop.schemas import (
    BuildState,
    ContractState,
    EnvironmentState,
    PolicyState,
)


class TestTypedState:
    """BuildState fields must be typed models, not raw dicts."""

    def test_contract_state_is_typed(self):
        contract = BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
        )
        state = BuildState(
            contract=ContractState(data=contract),
        )
        assert state.contract is not None
        assert state.contract.data.project_name == "test"
        assert state.contract.data.goals == ["test"]

    def test_environment_state_is_typed(self):
        env = EnvironmentSnapshot(
            os_name="Darwin", arch="arm64",
            python_version="3.12", python_path="/usr/bin/python3",
        )
        state = BuildState(
            environment=EnvironmentState(data=env),
        )
        assert state.environment is not None
        assert state.environment.data.os_name == "Darwin"

    def test_policy_state_is_typed(self):
        policy = PolicyDecision(autonomy_mode=AutonomyMode.PROCEED)
        state = BuildState(
            policy=PolicyState(data=policy),
        )
        assert state.policy is not None
        assert state.policy.data.autonomy_mode == AutonomyMode.PROCEED

    def test_state_serializes_and_deserializes(self):
        """State must survive JSON round-trip with typed fields intact."""
        contract = BuildContract(
            project_name="roundtrip", summary="test",
            goals=["survive serialization"], acceptance_criteria=["test"],
        )
        env = EnvironmentSnapshot(
            os_name="Linux", arch="x86_64",
            python_version="3.11", python_path="/usr/bin/python3",
        )
        policy = PolicyDecision(
            autonomy_mode=AutonomyMode.CHECKPOINT,
            reasons=["missing secret"],
        )

        state = BuildState(
            idea="test idea",
            contract=ContractState(data=contract),
            environment=EnvironmentState(data=env),
            policy=PolicyState(data=policy),
        )

        # Round-trip through JSON
        json_str = state.model_dump_json()
        restored = BuildState.model_validate_json(json_str)

        assert restored.contract.data.project_name == "roundtrip"
        assert restored.environment.data.os_name == "Linux"
        assert restored.policy.data.autonomy_mode == AutonomyMode.CHECKPOINT
        assert restored.policy.data.reasons == ["missing secret"]


class TestStateSchemaVersion:
    """BuildState schema_version must be Literal['1']."""

    def test_default_version(self):
        state = BuildState()
        assert state.schema_version == "1"

    def test_wrong_version_fails(self):
        with pytest.raises(ValidationError):
            BuildState(schema_version="99")

    def test_round_trip_preserves_version(self):
        state = BuildState(idea="test")
        json_str = state.model_dump_json()
        restored = BuildState.model_validate_json(json_str)
        assert restored.schema_version == "1"
