"""Tests for BuildContract schema validation."""

import pytest
from pydantic import ValidationError

from build_loop.contract import (
    SCHEMA_VERSION,
    BehavioralExpectation,
    BuildContract,
    CliExitSignal,
    FileExistsSignal,
    HttpProbeSignal,
    ImportCheckSignal,
    Invariant,
    StdoutContainsSignal,
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

    def test_schema_version_is_literal(self):
        """schema_version must be exactly '1' — other values fail validation."""
        c = BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
        )
        assert c.schema_version == "1"

    def test_rejects_wrong_schema_version(self):
        with pytest.raises(ValidationError):
            BuildContract(
                schema_version="99",
                project_name="test", summary="test",
                goals=["test"], acceptance_criteria=["test"],
            )

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
                CliExitSignal(
                    description="Script runs without error",
                    command="python",
                    args=["main.py", "--dry-run"],
                    expect_exit=0,
                ),
                FileExistsSignal(
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
        assert c.success_signals[0].type == "cli_exit"
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
        with pytest.raises(ValidationError):
            BuildContract(
                project_name="test", summary="test",
                goals=["test"], acceptance_criteria=["test"],
                run_mode="invalid",
            )


class TestSuccessSignalDiscrimination:
    """Each signal type must require its own fields and reject others."""

    def test_cli_exit_requires_command(self):
        with pytest.raises(ValidationError):
            CliExitSignal(description="test")  # missing command

    def test_http_probe_requires_path(self):
        with pytest.raises(ValidationError):
            HttpProbeSignal(description="test")  # missing path

    def test_file_exists_requires_file_path(self):
        with pytest.raises(ValidationError):
            FileExistsSignal(description="test")  # missing file_path

    def test_import_check_requires_module_name(self):
        with pytest.raises(ValidationError):
            ImportCheckSignal(description="test")  # missing module_name

    def test_stdout_contains_requires_fields(self):
        with pytest.raises(ValidationError):
            StdoutContainsSignal(description="test")  # missing command & expect_contains

    def test_valid_cli_exit(self):
        s = CliExitSignal(description="runs", command="python", args=["main.py"])
        assert s.type == "cli_exit"
        assert s.expect_exit == 0

    def test_valid_http_probe(self):
        s = HttpProbeSignal(description="health", path="/health")
        assert s.type == "http_probe"
        assert s.expect_status == 200

    def test_discriminated_union_in_contract(self):
        """Signals are discriminated by 'type' field in the contract."""
        c = BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            success_signals=[
                {"type": "cli_exit", "description": "runs", "command": "python"},
                {"type": "http_probe", "description": "health", "path": "/health"},
                {"type": "file_exists", "description": "output", "file_path": "out.csv"},
            ],
        )
        assert isinstance(c.success_signals[0], CliExitSignal)
        assert isinstance(c.success_signals[1], HttpProbeSignal)
        assert isinstance(c.success_signals[2], FileExistsSignal)

    def test_invalid_signal_type_in_contract(self):
        with pytest.raises(ValidationError):
            BuildContract(
                project_name="test", summary="test",
                goals=["test"], acceptance_criteria=["test"],
                success_signals=[
                    {"type": "nonexistent", "description": "bad"},
                ],
            )


class TestExtraFieldsRejected:
    """extra='forbid' must reject unknown fields on all contract models."""

    def test_contract_rejects_unknown_field(self):
        with pytest.raises(ValidationError, match="extra"):
            BuildContract(
                project_name="test", summary="test",
                goals=["test"], acceptance_criteria=["test"],
                totally_bogus_field="surprise",
            )

    def test_invariant_rejects_unknown_field(self):
        with pytest.raises(ValidationError, match="extra"):
            Invariant(description="test", severity="high")

    def test_behavioral_expectation_rejects_unknown_field(self):
        with pytest.raises(ValidationError, match="extra"):
            BehavioralExpectation(
                description="test", given="x", expect="y",
                priority="high",
            )

    def test_cli_exit_signal_rejects_unknown_field(self):
        with pytest.raises(ValidationError, match="extra"):
            CliExitSignal(
                description="test", command="python",
                path="/health",  # wrong field for cli_exit
            )

    def test_http_probe_rejects_unknown_field(self):
        with pytest.raises(ValidationError, match="extra"):
            HttpProbeSignal(
                description="test", path="/health",
                command="python",  # wrong field for http_probe
            )
