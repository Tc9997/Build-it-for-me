"""Tests for BuildContract schema validation."""

import pytest

from build_loop.contract import (
    SCHEMA_VERSION,
    BehavioralExpectation,
    BuildContract,
    Invariant,
    SuccessSignal,
    SuccessSignalType,
)


class TestBuildContract:
    """BuildContract must enforce structure and carry schema version."""

    def test_minimal_valid_contract(self):
        c = BuildContract(
            project_name="test",
            summary="A test project",
            goals=["do the thing"],
            acceptance_criteria=["thing is done"],
        )
        assert c.schema_version == SCHEMA_VERSION
        assert c.run_mode == "batch"
        assert c.non_goals == []
        assert c.invariants == []

    def test_schema_version_is_explicit(self):
        c = BuildContract(
            project_name="test",
            summary="test",
            goals=["test"],
            acceptance_criteria=["test"],
        )
        assert c.schema_version == "1"

    def test_full_contract_with_signals(self):
        c = BuildContract(
            project_name="wine_scraper",
            summary="Scrapes wine auctions",
            goals=["Scrape auction listings", "Parse lot data"],
            non_goals=["Build a recommendation engine"],
            constraints=["Python 3.12", "No Selenium"],
            target_runtime="python3.12",
            run_mode="batch",
            external_dependencies=["acker.com website"],
            secrets_required=[],
            acceptance_criteria=["Outputs CSV with lot data"],
            success_signals=[
                SuccessSignal(
                    type=SuccessSignalType.CLI_EXIT,
                    description="Script runs without error",
                    command="python",
                    args=["main.py", "--dry-run"],
                    expect_exit=0,
                ),
                SuccessSignal(
                    type=SuccessSignalType.FILE_EXISTS,
                    description="Output CSV is created",
                    file_path="output.csv",
                ),
            ],
            behavioral_expectations=[
                BehavioralExpectation(
                    description="Parses auction page HTML",
                    given="A sample auction HTML page",
                    expect="Extracts lot number, title, estimate, and current bid",
                ),
            ],
            invariants=[
                Invariant(
                    description="No hardcoded URLs that bypass robots.txt",
                    category="security",
                ),
            ],
        )
        assert len(c.success_signals) == 2
        assert c.success_signals[0].type == SuccessSignalType.CLI_EXIT
        assert len(c.invariants) == 1
        assert c.non_goals == ["Build a recommendation engine"]

    def test_service_mode(self):
        c = BuildContract(
            project_name="api_server",
            summary="A REST API",
            goals=["Serve /health endpoint"],
            acceptance_criteria=["GET /health returns 200"],
            run_mode="service",
        )
        assert c.run_mode == "service"

    def test_rejects_invalid_run_mode(self):
        with pytest.raises(Exception):
            BuildContract(
                project_name="test",
                summary="test",
                goals=["test"],
                acceptance_criteria=["test"],
                run_mode="invalid",
            )
