"""Tests for freeform iteration journal: issue extraction, persistence, backward compat.

Covers:
- FreeformJournal model behavior (record, filter, summarize)
- Issue extraction for each failure type (integration, setup, test, acceptance, crash)
- Journal persistence via BuildState serialization round-trip
- Backward compatibility: old state JSON without freeform_journal still loads
- FreeformOrchestrator wires journal to state correctly
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from build_loop.common.pipeline import PipelineError
from build_loop.freeform_journal import (
    AttemptRecord,
    FreeformIssue,
    FreeformJournal,
    IssueKind,
    IssueSeverity,
    IssueSource,
)
from build_loop.schemas import BuildState


# =========================================================================
# Journal model basics
# =========================================================================

class TestFreeformJournalModel:
    def test_empty_journal(self):
        j = FreeformJournal()
        assert j.issues == []
        assert j.attempts == []
        assert not j.has_blocking
        assert j.blocking_issues == []
        assert j.retryable_issues == []

    def test_record_issue(self):
        j = FreeformJournal()
        issue = j.record_issue(
            IssueKind.TEST_FAILURE,
            IssueSource.TEST,
            IssueSeverity.DEGRADED,
            "Tests failed after 3 rounds",
            detail="AssertionError in test_foo",
            retryable=True,
            command="pytest",
        )
        assert len(j.issues) == 1
        assert issue is j.issues[0]
        assert issue.kind == IssueKind.TEST_FAILURE
        assert issue.retryable is True
        assert issue.command == "pytest"

    def test_record_attempt(self):
        j = FreeformJournal()
        attempt = j.record_attempt(
            "setup", success=True, summary="pip install ok",
            command="pip install -r requirements.txt", exit_code=0,
        )
        assert len(j.attempts) == 1
        assert attempt.phase == "setup"
        assert attempt.success is True
        assert attempt.exit_code == 0

    def test_blocking_filter(self):
        j = FreeformJournal()
        j.record_issue(IssueKind.SETUP_FAILURE, IssueSource.SETUP, IssueSeverity.BLOCKING, "blocked")
        j.record_issue(IssueKind.TEST_FAILURE, IssueSource.TEST, IssueSeverity.DEGRADED, "degraded")
        assert len(j.blocking_issues) == 1
        assert j.has_blocking
        assert j.blocking_issues[0].summary == "blocked"

    def test_retryable_filter(self):
        j = FreeformJournal()
        j.record_issue(IssueKind.INTEGRATION_FAILURE, IssueSource.INTEGRATE, IssueSeverity.BLOCKING, "retry me", retryable=True)
        j.record_issue(IssueKind.UNEXPECTED_CRASH, IssueSource.BUILD, IssueSeverity.BLOCKING, "no retry", retryable=False)
        assert len(j.retryable_issues) == 1
        assert j.retryable_issues[0].summary == "retry me"

    def test_summary_clean_run(self):
        j = FreeformJournal()
        j.record_attempt("research", success=True)
        j.record_attempt("plan", success=True)
        s = j.summary_text()
        assert "no issues" in s.lower()
        assert "2 phase" in s

    def test_summary_with_issues(self):
        j = FreeformJournal()
        j.record_issue(IssueKind.TEST_FAILURE, IssueSource.TEST, IssueSeverity.BLOCKING, "fail", retryable=True)
        j.record_attempt("test", success=False)
        s = j.summary_text()
        assert "1 issue" in s
        assert "1 blocking" in s
        assert "1 retryable" in s

    def test_issue_timestamps_populated(self):
        j = FreeformJournal()
        issue = j.record_issue(IssueKind.PIPELINE_ERROR, IssueSource.BUILD, IssueSeverity.BLOCKING, "err")
        assert issue.timestamp  # non-empty
        assert "T" in issue.timestamp  # ISO format

    def test_attempt_number(self):
        j = FreeformJournal()
        a1 = j.record_attempt("test", success=False, attempt_number=1)
        a2 = j.record_attempt("test", success=True, attempt_number=2)
        assert a1.attempt_number == 1
        assert a2.attempt_number == 2

    def test_schema_version(self):
        j = FreeformJournal()
        assert j.schema_version == "1"


class TestFreeformIssueValidation:
    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            FreeformIssue(
                kind=IssueKind.TEST_FAILURE,
                source=IssueSource.TEST,
                severity=IssueSeverity.DEGRADED,
                summary="test",
                bogus_field="bad",
            )

    def test_attempt_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            AttemptRecord(phase="test", bogus="bad")

    def test_attempt_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            AttemptRecord(phase="test", attempt_number=0)


# =========================================================================
# Issue extraction per failure type
# =========================================================================

class TestIssueExtraction:
    """Verifies that each failure type produces the correct issue kind."""

    def test_integration_failure_issue(self):
        j = FreeformJournal()
        j.record_issue(
            IssueKind.INTEGRATION_FAILURE,
            IssueSource.INTEGRATE,
            IssueSeverity.BLOCKING,
            "Integration failed: circular imports",
            retryable=True,
        )
        assert j.issues[0].kind == IssueKind.INTEGRATION_FAILURE
        assert j.issues[0].retryable is True
        assert j.has_blocking

    def test_setup_failure_issue(self):
        j = FreeformJournal()
        j.record_issue(
            IssueKind.SETUP_FAILURE,
            IssueSource.SETUP,
            IssueSeverity.BLOCKING,
            "pip install failed",
            command="pip install -r requirements.txt",
            retryable=True,
        )
        assert j.issues[0].kind == IssueKind.SETUP_FAILURE
        assert j.issues[0].command == "pip install -r requirements.txt"

    def test_test_failure_issue(self):
        j = FreeformJournal()
        j.record_issue(
            IssueKind.TEST_FAILURE,
            IssueSource.TEST,
            IssueSeverity.DEGRADED,
            "Tests did not pass after debug loop",
            detail="FAILED test_main.py::test_output - AssertionError",
            command="pytest",
            retryable=True,
        )
        assert j.issues[0].kind == IssueKind.TEST_FAILURE
        assert j.issues[0].severity == IssueSeverity.DEGRADED

    def test_acceptance_failure_issue(self):
        j = FreeformJournal()
        j.record_issue(
            IssueKind.ACCEPTANCE_FAILURE,
            IssueSource.ACCEPTANCE,
            IssueSeverity.BLOCKING,
            "Acceptance verdict: fail",
            verdict="fail",
            retryable=False,
        )
        assert j.issues[0].kind == IssueKind.ACCEPTANCE_FAILURE
        assert j.issues[0].verdict == "fail"
        assert j.issues[0].retryable is False

    def test_acceptance_incomplete_issue(self):
        j = FreeformJournal()
        j.record_issue(
            IssueKind.ACCEPTANCE_INCOMPLETE,
            IssueSource.ACCEPTANCE,
            IssueSeverity.DEGRADED,
            "Acceptance verdict: incomplete",
            verdict="incomplete",
            retryable=True,
        )
        assert j.issues[0].kind == IssueKind.ACCEPTANCE_INCOMPLETE
        assert j.issues[0].severity == IssueSeverity.DEGRADED

    def test_pipeline_error_issue(self):
        j = FreeformJournal()
        j.record_issue(
            IssueKind.PIPELINE_ERROR,
            IssueSource.BUILD,
            IssueSeverity.BLOCKING,
            "Module rejected after 3 review rounds",
            retryable=True,
        )
        assert j.issues[0].kind == IssueKind.PIPELINE_ERROR

    def test_unexpected_crash_issue(self):
        j = FreeformJournal()
        j.record_issue(
            IssueKind.UNEXPECTED_CRASH,
            IssueSource.RESEARCH,
            IssueSeverity.BLOCKING,
            "KeyError: 'missing_key'",
            retryable=False,
        )
        assert j.issues[0].kind == IssueKind.UNEXPECTED_CRASH
        assert j.issues[0].retryable is False


# =========================================================================
# Persistence / serialization
# =========================================================================

class TestJournalPersistence:
    """Journal must survive BuildState JSON round-trip."""

    def test_journal_round_trip(self):
        j = FreeformJournal()
        j.record_issue(
            IssueKind.TEST_FAILURE,
            IssueSource.TEST,
            IssueSeverity.DEGRADED,
            "tests failed",
            command="pytest",
            retryable=True,
        )
        j.record_attempt("test", success=False, command="pytest", exit_code=1)
        j.record_attempt("test", success=True, attempt_number=2, command="pytest", exit_code=0)

        state = BuildState(idea="round trip test", freeform_journal=j)
        json_str = state.model_dump_json()
        restored = BuildState.model_validate_json(json_str)

        assert restored.freeform_journal is not None
        rj = restored.freeform_journal
        assert len(rj.issues) == 1
        assert rj.issues[0].kind == IssueKind.TEST_FAILURE
        assert rj.issues[0].retryable is True
        assert len(rj.attempts) == 2
        assert rj.attempts[0].success is False
        assert rj.attempts[1].success is True
        assert rj.attempts[1].attempt_number == 2

    def test_journal_none_round_trip(self):
        """Template-first state has freeform_journal=None and round-trips fine."""
        state = BuildState(idea="template project")
        assert state.freeform_journal is None
        json_str = state.model_dump_json()
        restored = BuildState.model_validate_json(json_str)
        assert restored.freeform_journal is None

    def test_journal_preserves_all_issue_fields(self):
        j = FreeformJournal()
        j.record_issue(
            IssueKind.ACCEPTANCE_FAILURE,
            IssueSource.ACCEPTANCE,
            IssueSeverity.BLOCKING,
            "fail",
            detail="did not meet criteria",
            verdict="fail",
            command="smoke test",
            retryable=False,
        )
        state = BuildState(freeform_journal=j)
        restored = BuildState.model_validate_json(state.model_dump_json())
        issue = restored.freeform_journal.issues[0]
        assert issue.detail == "did not meet criteria"
        assert issue.verdict == "fail"
        assert issue.command == "smoke test"
        assert issue.timestamp  # preserved


# =========================================================================
# Backward compatibility
# =========================================================================

class TestBackwardCompatibility:
    """Old saved state JSON (without freeform_journal) must still load."""

    def test_old_state_without_journal_loads(self):
        """Simulate loading a state file from before the journal was added."""
        old_state_dict = {
            "schema_version": "1",
            "idea": "old project",
            "output_dir": "/tmp/old",
        }
        state = BuildState.model_validate(old_state_dict)
        assert state.idea == "old project"
        assert state.freeform_journal is None

    def test_old_state_json_without_journal_loads(self):
        """Same test but via JSON string (the actual persistence path)."""
        old_json = json.dumps({
            "schema_version": "1",
            "idea": "old project from json",
            "output_dir": "/tmp/old",
        })
        state = BuildState.model_validate_json(old_json)
        assert state.idea == "old project from json"
        assert state.freeform_journal is None

    def test_old_state_with_all_fields_but_no_journal(self):
        """A full old state (with acceptance, verification, etc.) still loads."""
        old_state_dict = {
            "schema_version": "1",
            "idea": "full old project",
            "output_dir": "/tmp/old",
            "debug_rounds": 3,
            "optimization_count": 1,
            "acceptance": {
                "verdict": "pass",
                "criteria_checked": ["runs"],
                "criteria_passed": ["runs"],
                "criteria_failed": [],
                "notes": "all good",
            },
        }
        state = BuildState.model_validate(old_state_dict)
        assert state.acceptance is not None
        assert state.acceptance.verdict.value == "pass"
        assert state.freeform_journal is None


# =========================================================================
# Orchestrator wiring
# =========================================================================

class TestOrchestratorWiring:
    """FreeformOrchestrator initializes journal correctly."""

    def test_orchestrator_has_journal(self):
        from build_loop.modes.freeform import FreeformOrchestrator
        orch = FreeformOrchestrator(output_dir="/tmp/test-journal")
        assert orch.journal is not None
        assert isinstance(orch.journal, FreeformJournal)
        assert orch.state.freeform_journal is orch.journal

    def test_orchestrator_journal_is_empty_at_start(self):
        from build_loop.modes.freeform import FreeformOrchestrator
        orch = FreeformOrchestrator(output_dir="/tmp/test-journal-empty")
        assert len(orch.journal.issues) == 0
        assert len(orch.journal.attempts) == 0

    def test_architect_freeform_has_journal(self):
        from build_loop.agents.architect import ArchitectAgent
        from build_loop.modes import BuildMode
        agent = ArchitectAgent(output_dir="/tmp/test-arch-journal", mode=BuildMode.FREEFORM)
        assert agent.state.freeform_journal is not None

    def test_architect_template_first_has_no_journal(self):
        from build_loop.agents.architect import ArchitectAgent
        from build_loop.modes import BuildMode
        agent = ArchitectAgent(output_dir="/tmp/test-arch-no-journal", mode=BuildMode.TEMPLATE_FIRST)
        assert agent.state.freeform_journal is None


# =========================================================================
# Enum coverage
# =========================================================================

# =========================================================================
# Orchestrator-level regression tests (P1/P2 bugs)
# =========================================================================

class TestOrchestratorJournalRecording:
    """Regression: the orchestrator must record the correct issue kind and
    source when pipeline functions raise exceptions."""

    def _make_orchestrator_through_phase(self, target_phase, tmp_path):
        """Create a FreeformOrchestrator with all agents mocked, run it,
        and let it reach the target phase before the injected failure."""
        from unittest.mock import MagicMock, patch
        from build_loop.modes.freeform import FreeformOrchestrator
        from build_loop.schemas import (
            BuildPlan, ModuleSpec, BuildArtifact, IntegrationResult,
            ResearchReport, ExecResult, TaskSize,
        )

        orch = FreeformOrchestrator(output_dir=str(tmp_path))

        # Stub all agents to return minimal valid results
        orch.researcher.run = MagicMock(return_value=ResearchReport(
            feasibility="feasible", recommended_stack=["pytest"],
        ))
        orch.planner.run = MagicMock(return_value=BuildPlan(
            project_name="test", description="test",
            modules=[ModuleSpec(id="m1", name="m1", description="mod", size=TaskSize.SMALL)],
            build_order=[["m1"]],
        ))

        return orch

    def test_test_debug_exhaustion_records_test_failure(self, tmp_path):
        """When test_and_debug_loop raises PipelineError (debug rounds exhausted),
        the journal must contain TEST_FAILURE, not PIPELINE_ERROR."""
        from unittest.mock import MagicMock, patch
        from build_loop.schemas import (
            BuildPlan, ModuleSpec, BuildArtifact, IntegrationResult,
            ResearchReport, ExecResult, TaskSize,
        )

        orch = self._make_orchestrator_through_phase("test", tmp_path)

        # We need to patch the pipeline functions that the orchestrator calls
        with patch("build_loop.modes.freeform.build_all") as mock_build, \
             patch("build_loop.modes.freeform.write_project") as mock_write, \
             patch("build_loop.modes.freeform.setup_environment") as mock_setup, \
             patch("build_loop.modes.freeform.test_and_debug_loop") as mock_test, \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            # Integration succeeds
            orch.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=["m1"], success=True,
            ))

            # test_and_debug_loop raises PipelineError (the bug trigger)
            mock_test.side_effect = PipelineError("Tests did not pass after 3 debug rounds")

            # Add a fake exec_history entry so the handler can extract details
            orch.state.exec_history.append(ExecResult(
                command="pytest -v", exit_code=1, stderr="FAILED test_main.py",
            ))

            orch.run("test idea")

        # The journal must have TEST_FAILURE, NOT PIPELINE_ERROR
        issue_kinds = [i.kind for i in orch.journal.issues]
        assert IssueKind.TEST_FAILURE in issue_kinds, (
            f"Expected TEST_FAILURE but got: {issue_kinds}"
        )
        assert IssueKind.PIPELINE_ERROR not in issue_kinds, (
            f"Should not have generic PIPELINE_ERROR, got: {issue_kinds}"
        )

        # Verify the issue details
        test_issue = [i for i in orch.journal.issues if i.kind == IssueKind.TEST_FAILURE][0]
        assert test_issue.source == IssueSource.TEST
        assert test_issue.severity == IssueSeverity.BLOCKING
        assert test_issue.retryable is True
        assert "3 debug rounds" in test_issue.summary

        # There should also be a failed test attempt record
        test_attempts = [a for a in orch.journal.attempts if a.phase == "test"]
        assert len(test_attempts) >= 1
        assert test_attempts[-1].success is False

    def test_optimize_failure_records_optimize_source(self, tmp_path):
        """When optimize() raises PipelineError after tests passed,
        the journal issue source must be OPTIMIZE, not TEST."""
        from unittest.mock import MagicMock, patch
        from build_loop.schemas import (
            BuildPlan, ModuleSpec, BuildArtifact, IntegrationResult,
            ResearchReport, ExecResult, TaskSize,
        )

        orch = self._make_orchestrator_through_phase("optimize", tmp_path)

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.setup_environment"), \
             patch("build_loop.modes.freeform.test_and_debug_loop"), \
             patch("build_loop.modes.freeform.optimize") as mock_optimize, \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            orch.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=["m1"], success=True,
            ))

            # Tests passed — exec_history has a successful test run
            orch.state.exec_history.append(ExecResult(
                command="pytest -v", exit_code=0, stdout="5 passed",
            ))

            # optimize raises PipelineError
            mock_optimize.side_effect = PipelineError(
                "Optimization broke tests and could not be repaired"
            )

            orch.run("test idea")

        # The journal must attribute the error to OPTIMIZE, not TEST
        issue_kinds = [i.kind for i in orch.journal.issues]
        assert IssueKind.PIPELINE_ERROR in issue_kinds

        pipeline_issue = [i for i in orch.journal.issues if i.kind == IssueKind.PIPELINE_ERROR][0]
        assert pipeline_issue.source == IssueSource.OPTIMIZE, (
            f"Expected source=OPTIMIZE but got source={pipeline_issue.source}"
        )
        assert "Optimization" in pipeline_issue.summary


class TestEnumValues:
    """All enum values are distinct and have expected members."""

    def test_issue_kinds(self):
        kinds = [k.value for k in IssueKind]
        assert len(kinds) == len(set(kinds))
        assert "integration_failure" in kinds
        assert "unexpected_crash" in kinds

    def test_severities(self):
        assert IssueSeverity.BLOCKING.value == "blocking"
        assert IssueSeverity.DEGRADED.value == "degraded"
        assert IssueSeverity.INFO.value == "info"

    def test_sources(self):
        sources = [s.value for s in IssueSource]
        assert "research" in sources
        assert "acceptance" in sources
        assert len(sources) == 9  # all pipeline phases
