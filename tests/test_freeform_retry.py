"""Tests for the freeform retry planner and orchestrator retry behavior.

Covers:
- Retry planner decision mapping for each issue kind
- Budget exhaustion
- Orchestrator-level: setup retry, test retry, budget exhaustion, successful retry
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from build_loop.common.pipeline import PipelineError
from build_loop.freeform_journal import (
    FreeformJournal,
    FreeformIssue,
    IssueKind,
    IssueSeverity,
    IssueSource,
)
from build_loop.freeform_retry import (
    RetryAction,
    RetryDecision,
    plan_retry,
)
from build_loop.schemas import (
    BuildPlan,
    BuildState,
    ExecResult,
    IntegrationResult,
    ModuleSpec,
    ResearchReport,
    TaskSize,
)


# =========================================================================
# Retry planner unit tests
# =========================================================================

class TestRetryPlannerMapping:
    """plan_retry must return correct action for each issue kind."""

    def _make_issue(self, kind: IssueKind, source: IssueSource) -> FreeformIssue:
        return FreeformIssue(
            kind=kind, source=source, severity=IssueSeverity.BLOCKING,
            summary="test", retryable=True,
        )

    def test_setup_failure_retries_once(self):
        j = FreeformJournal()
        issue = self._make_issue(IssueKind.SETUP_FAILURE, IssueSource.SETUP)
        j.issues.append(issue)
        j.record_attempt("setup", success=False)

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.RETRY_PHASE
        assert decision.phase == "setup"
        assert decision.budget_remaining == 0  # 1 retry allowed, will be 0 after

    def test_test_failure_retries_once(self):
        j = FreeformJournal()
        issue = self._make_issue(IssueKind.TEST_FAILURE, IssueSource.TEST)
        j.issues.append(issue)
        j.record_attempt("test", success=False)

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.RETRY_PHASE
        assert decision.phase == "test"
        assert decision.budget_remaining == 0

    def test_unexpected_crash_stops(self):
        j = FreeformJournal()
        issue = self._make_issue(IssueKind.UNEXPECTED_CRASH, IssueSource.BUILD)
        j.issues.append(issue)

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.STOP
        assert "not automatically retryable" in decision.rationale

    def test_acceptance_failure_stops(self):
        j = FreeformJournal()
        issue = self._make_issue(IssueKind.ACCEPTANCE_FAILURE, IssueSource.ACCEPTANCE)
        j.issues.append(issue)

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.STOP

    def test_acceptance_incomplete_stops(self):
        j = FreeformJournal()
        issue = self._make_issue(IssueKind.ACCEPTANCE_INCOMPLETE, IssueSource.ACCEPTANCE)
        j.issues.append(issue)

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.STOP

    def test_integration_failure_stops(self):
        j = FreeformJournal()
        issue = self._make_issue(IssueKind.INTEGRATION_FAILURE, IssueSource.INTEGRATE)
        j.issues.append(issue)

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.STOP

    def test_pipeline_error_stops(self):
        j = FreeformJournal()
        issue = self._make_issue(IssueKind.PIPELINE_ERROR, IssueSource.BUILD)
        j.issues.append(issue)

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.STOP


class TestRetryBudgetExhaustion:
    """Budget must be enforced: no infinite retries."""

    def _make_issue(self, kind: IssueKind, source: IssueSource) -> FreeformIssue:
        return FreeformIssue(
            kind=kind, source=source, severity=IssueSeverity.BLOCKING,
            summary="test", retryable=True,
        )

    def test_setup_budget_exhausted_after_two_attempts(self):
        j = FreeformJournal()
        j.record_attempt("setup", success=False, attempt_number=1)
        j.record_attempt("setup", success=False, attempt_number=2)
        issue = self._make_issue(IssueKind.SETUP_FAILURE, IssueSource.SETUP)
        j.issues.append(issue)

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.STOP
        assert "exhausted" in decision.rationale.lower()
        assert decision.budget_remaining == 0

    def test_test_budget_exhausted_after_two_attempts(self):
        j = FreeformJournal()
        j.record_attempt("test", success=False, attempt_number=1)
        j.record_attempt("test", success=False, attempt_number=2)
        issue = self._make_issue(IssueKind.TEST_FAILURE, IssueSource.TEST)
        j.issues.append(issue)

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.STOP
        assert decision.budget_remaining == 0

    def test_first_attempt_has_budget(self):
        """Before any attempts, budget should allow a retry."""
        j = FreeformJournal()
        issue = self._make_issue(IssueKind.SETUP_FAILURE, IssueSource.SETUP)
        j.issues.append(issue)
        # No attempts recorded yet — first failure

        decision = plan_retry(issue, j)
        assert decision.action == RetryAction.RETRY_PHASE
        assert decision.budget_remaining == 1  # initial + 1 retry = 2 total, 0 used


class TestRetryDecisionModel:
    def test_extra_fields_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RetryDecision(
                action=RetryAction.STOP, phase="test",
                rationale="x", budget_remaining=0, bogus="bad",
            )

    def test_negative_budget_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RetryDecision(
                action=RetryAction.STOP, phase="test",
                rationale="x", budget_remaining=-1,
            )


# =========================================================================
# Orchestrator-level regression tests
# =========================================================================

def _make_patched_orchestrator(tmp_path):
    """Create a FreeformOrchestrator with agents mocked, ready to run."""
    from build_loop.modes.freeform import FreeformOrchestrator

    orch = FreeformOrchestrator(output_dir=str(tmp_path))
    orch.researcher.run = MagicMock(return_value=ResearchReport(
        feasibility="feasible", recommended_stack=["pytest"],
    ))
    orch.planner.run = MagicMock(return_value=BuildPlan(
        project_name="test", description="test",
        modules=[ModuleSpec(id="m1", name="m1", description="mod", size=TaskSize.SMALL)],
        build_order=[["m1"]],
    ))
    orch.integrator.run = MagicMock(return_value=IntegrationResult(
        modules_integrated=["m1"], success=True,
    ))
    return orch


class TestOrchestratorSetupRetry:
    """Setup failure must trigger exactly one targeted retry."""

    def test_setup_failure_retries_then_stops(self, tmp_path):
        """Setup fails twice -> pipeline stops with 2 setup attempts."""
        orch = _make_patched_orchestrator(tmp_path)

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.setup_environment") as mock_setup, \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            mock_setup.side_effect = PipelineError("pip install failed")
            orch.run("test idea")

        # Should have exactly 2 setup attempts (initial + 1 retry)
        setup_attempts = [a for a in orch.journal.attempts if a.phase == "setup"]
        assert len(setup_attempts) == 2
        assert setup_attempts[0].success is False
        assert setup_attempts[0].attempt_number == 1
        assert setup_attempts[1].success is False
        assert setup_attempts[1].attempt_number == 2

        # Should have 2 SETUP_FAILURE issues
        setup_issues = [i for i in orch.journal.issues if i.kind == IssueKind.SETUP_FAILURE]
        assert len(setup_issues) == 2

    def test_setup_retry_succeeds_continues_pipeline(self, tmp_path):
        """Setup fails once, succeeds on retry -> pipeline continues to test."""
        orch = _make_patched_orchestrator(tmp_path)
        setup_call_count = 0

        def setup_side_effect(*args, **kwargs):
            nonlocal setup_call_count
            setup_call_count += 1
            if setup_call_count == 1:
                raise PipelineError("pip install failed first time")
            # Second call succeeds

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.setup_environment") as mock_setup, \
             patch("build_loop.modes.freeform.test_and_debug_loop") as mock_test, \
             patch("build_loop.modes.freeform.optimize"), \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            mock_setup.side_effect = setup_side_effect

            # Acceptance — need to mock to avoid hitting real agents
            orch.acceptance.run = MagicMock(return_value=MagicMock(
                verdict=MagicMock(value="pass"),
                notes="",
            ))

            orch.run("test idea")

        # Setup: attempt 1 failed, attempt 2 succeeded
        setup_attempts = [a for a in orch.journal.attempts if a.phase == "setup"]
        assert len(setup_attempts) == 2
        assert setup_attempts[0].success is False
        assert setup_attempts[1].success is True
        assert setup_attempts[1].summary == "Setup succeeded on retry"

        # Pipeline continued — test phase was reached
        assert mock_test.called


class TestOrchestratorTestRetry:
    """Test failure must trigger exactly one targeted retry."""

    def test_test_failure_retries_then_stops(self, tmp_path):
        """test_and_debug_loop fails twice -> pipeline stops with 2 test attempts."""
        orch = _make_patched_orchestrator(tmp_path)

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.setup_environment"), \
             patch("build_loop.modes.freeform.test_and_debug_loop") as mock_test, \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            mock_test.side_effect = PipelineError("Tests did not pass after 3 debug rounds")
            # Add exec_history for detail extraction
            orch.state.exec_history.append(ExecResult(
                command="pytest -v", exit_code=1, stderr="FAILED test_main.py",
            ))

            orch.run("test idea")

        # Should have exactly 2 test attempts
        test_attempts = [a for a in orch.journal.attempts if a.phase == "test"]
        assert len(test_attempts) == 2
        assert test_attempts[0].success is False
        assert test_attempts[1].success is False

        # Should have 2 TEST_FAILURE issues
        test_issues = [i for i in orch.journal.issues if i.kind == IssueKind.TEST_FAILURE]
        assert len(test_issues) == 2

        # Should NOT have generic PIPELINE_ERROR
        pipeline_errors = [i for i in orch.journal.issues if i.kind == IssueKind.PIPELINE_ERROR]
        assert len(pipeline_errors) == 0

    def test_test_retry_succeeds_continues_pipeline(self, tmp_path):
        """test_and_debug_loop fails once, succeeds on retry -> pipeline continues."""
        orch = _make_patched_orchestrator(tmp_path)
        test_call_count = 0

        def test_side_effect(*args, **kwargs):
            nonlocal test_call_count
            test_call_count += 1
            if test_call_count == 1:
                raise PipelineError("Tests did not pass after 3 debug rounds")

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.setup_environment"), \
             patch("build_loop.modes.freeform.test_and_debug_loop") as mock_test, \
             patch("build_loop.modes.freeform.optimize") as mock_optimize, \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            mock_test.side_effect = test_side_effect
            orch.state.exec_history.append(ExecResult(
                command="pytest -v", exit_code=1, stderr="FAILED",
            ))

            orch.acceptance.run = MagicMock(return_value=MagicMock(
                verdict=MagicMock(value="pass"),
                notes="",
            ))

            orch.run("test idea")

        # Test: attempt 1 failed, attempt 2 succeeded
        test_attempts = [a for a in orch.journal.attempts if a.phase == "test"]
        assert len(test_attempts) == 2
        assert test_attempts[0].success is False
        assert test_attempts[1].success is True
        assert "retry" in test_attempts[1].summary.lower()

        # Pipeline continued to optimize
        assert mock_optimize.called

    def test_debug_rounds_reset_on_retry(self, tmp_path):
        """debug_rounds must be reset to 0 before the retry attempt."""
        orch = _make_patched_orchestrator(tmp_path)
        debug_rounds_at_retry = []

        def test_side_effect(state, *args, **kwargs):
            debug_rounds_at_retry.append(state.debug_rounds)
            if len(debug_rounds_at_retry) == 1:
                state.debug_rounds = 3  # simulate exhaustion
                raise PipelineError("Tests did not pass")

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.setup_environment"), \
             patch("build_loop.modes.freeform.test_and_debug_loop") as mock_test, \
             patch("build_loop.modes.freeform.optimize"), \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            mock_test.side_effect = test_side_effect
            orch.acceptance.run = MagicMock(return_value=MagicMock(
                verdict=MagicMock(value="pass"), notes="",
            ))

            orch.run("test idea")

        # First call: debug_rounds was 0 (initial)
        # Second call: debug_rounds should be 0 (reset before retry)
        assert len(debug_rounds_at_retry) == 2
        assert debug_rounds_at_retry[1] == 0


class TestOrchestratorNonRetryableStops:
    """Non-retryable failures must not trigger any retry."""

    def test_integration_failure_no_retry(self, tmp_path):
        orch = _make_patched_orchestrator(tmp_path)
        # Override integrator to fail
        orch.integrator.run = MagicMock(return_value=IntegrationResult(
            modules_integrated=[], success=False, issues=["circular import"],
        ))

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):
            orch.run("test idea")

        # Only 1 integrate attempt (no retry)
        integrate_attempts = [a for a in orch.journal.attempts if a.phase == "integrate"]
        assert len(integrate_attempts) == 1
        assert integrate_attempts[0].success is False


# =========================================================================
# P1 regression: recovered retry must not suppress later failures
# =========================================================================

class TestRecoveredRetryDoesNotSuppressLaterFailure:
    """Setup retries + later optimize failure must both appear in journal."""

    def test_setup_retry_succeeds_then_optimize_fails(self, tmp_path):
        """Setup fails once, retries successfully, then optimize raises
        PipelineError. Journal must contain both the setup failure AND
        a PIPELINE_ERROR with source=OPTIMIZE."""
        orch = _make_patched_orchestrator(tmp_path)
        setup_call_count = 0

        def setup_side_effect(*args, **kwargs):
            nonlocal setup_call_count
            setup_call_count += 1
            if setup_call_count == 1:
                raise PipelineError("pip install failed")

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.setup_environment") as mock_setup, \
             patch("build_loop.modes.freeform.test_and_debug_loop"), \
             patch("build_loop.modes.freeform.optimize") as mock_optimize, \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            mock_setup.side_effect = setup_side_effect
            mock_optimize.side_effect = PipelineError(
                "Optimization broke tests and could not be repaired"
            )

            orch.run("test idea")

        # Must have the earlier setup failure
        setup_issues = [i for i in orch.journal.issues if i.kind == IssueKind.SETUP_FAILURE]
        assert len(setup_issues) == 1

        # Must ALSO have the optimize failure as PIPELINE_ERROR
        pipeline_errors = [i for i in orch.journal.issues if i.kind == IssueKind.PIPELINE_ERROR]
        assert len(pipeline_errors) == 1, (
            f"Expected 1 PIPELINE_ERROR for optimize, got {len(pipeline_errors)}. "
            f"All issues: {[(i.kind.value, i.source.value) for i in orch.journal.issues]}"
        )
        assert pipeline_errors[0].source == IssueSource.OPTIMIZE
        assert "Optimization" in pipeline_errors[0].summary

    def test_test_retry_succeeds_then_acceptance_crash(self, tmp_path):
        """Test fails once, retries successfully, then acceptance crashes.
        Journal must contain both the test failure AND the crash."""
        orch = _make_patched_orchestrator(tmp_path)
        test_call_count = 0

        def test_side_effect(*args, **kwargs):
            nonlocal test_call_count
            test_call_count += 1
            if test_call_count == 1:
                raise PipelineError("Tests did not pass")

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.setup_environment"), \
             patch("build_loop.modes.freeform.test_and_debug_loop") as mock_test, \
             patch("build_loop.modes.freeform.optimize"), \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            mock_test.side_effect = test_side_effect
            # Acceptance agent crashes with a non-PipelineError
            orch.acceptance.run = MagicMock(
                side_effect=RuntimeError("LLM timeout")
            )

            orch.run("test idea")

        # Must have the earlier test failure
        test_issues = [i for i in orch.journal.issues if i.kind == IssueKind.TEST_FAILURE]
        assert len(test_issues) == 1

        # Must ALSO have the crash
        crash_issues = [i for i in orch.journal.issues if i.kind == IssueKind.UNEXPECTED_CRASH]
        assert len(crash_issues) == 1
        assert crash_issues[0].source == IssueSource.ACCEPTANCE


# =========================================================================
# P2 regression: retryable flag aligned with retry planner
# =========================================================================

class TestRetryableFlagAlignment:
    """Issues that plan_retry returns STOP for must not be retryable=True."""

    def test_integration_failure_not_retryable(self, tmp_path):
        """INTEGRATION_FAILURE must have retryable=False in the journal."""
        orch = _make_patched_orchestrator(tmp_path)
        orch.integrator.run = MagicMock(return_value=IntegrationResult(
            modules_integrated=[], success=False, issues=["circular import"],
        ))

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):
            orch.run("test idea")

        int_issues = [i for i in orch.journal.issues if i.kind == IssueKind.INTEGRATION_FAILURE]
        assert len(int_issues) == 1
        assert int_issues[0].retryable is False
        assert int_issues[0] not in orch.journal.retryable_issues

    def test_acceptance_incomplete_not_retryable(self):
        """ACCEPTANCE_INCOMPLETE must have retryable=False in the journal model."""
        from build_loop.freeform_journal import FreeformIssue
        issue = FreeformIssue(
            kind=IssueKind.ACCEPTANCE_INCOMPLETE,
            source=IssueSource.ACCEPTANCE,
            severity=IssueSeverity.DEGRADED,
            summary="incomplete",
            retryable=False,
        )
        j = FreeformJournal()
        j.issues.append(issue)
        assert issue not in j.retryable_issues

    def test_setup_failure_is_retryable(self):
        """SETUP_FAILURE should still be retryable=True (sanity check)."""
        j = FreeformJournal()
        issue = j.record_issue(
            IssueKind.SETUP_FAILURE, IssueSource.SETUP,
            IssueSeverity.BLOCKING, "pip failed", retryable=True,
        )
        assert issue in j.retryable_issues

    def test_test_failure_is_retryable(self):
        """TEST_FAILURE should still be retryable=True (sanity check)."""
        j = FreeformJournal()
        issue = j.record_issue(
            IssueKind.TEST_FAILURE, IssueSource.TEST,
            IssueSeverity.BLOCKING, "tests failed", retryable=True,
        )
        assert issue in j.retryable_issues


# =========================================================================
# Regression: reused orchestrator must not leak state across runs
# =========================================================================

class TestOrchestratorReuse:
    """Reusing the same FreeformOrchestrator for multiple runs must not
    leak _exception_already_journaled or stale journal entries."""

    def test_second_run_journals_new_failure(self, tmp_path):
        """Run 1: integration failure. Run 2: optimize failure.
        Run 2's journal must contain PIPELINE_ERROR(source=OPTIMIZE),
        not be suppressed by run 1's stale flag."""
        orch = _make_patched_orchestrator(tmp_path)

        # --- Run 1: integration failure ---
        orch.integrator.run = MagicMock(return_value=IntegrationResult(
            modules_integrated=[], success=False, issues=["circular import"],
        ))

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):
            orch.run("run 1")

        # Sanity: run 1 journaled the integration failure
        assert any(i.kind == IssueKind.INTEGRATION_FAILURE for i in orch.journal.issues)

        # --- Run 2: integration succeeds, optimize fails ---
        orch.integrator.run = MagicMock(return_value=IntegrationResult(
            modules_integrated=["m1"], success=True,
        ))

        with patch("build_loop.modes.freeform.build_all"), \
             patch("build_loop.modes.freeform.write_project"), \
             patch("build_loop.modes.freeform.setup_environment"), \
             patch("build_loop.modes.freeform.test_and_debug_loop"), \
             patch("build_loop.modes.freeform.optimize") as mock_opt, \
             patch("build_loop.modes.freeform.save_state"), \
             patch("build_loop.modes.freeform.phase"), \
             patch("build_loop.modes.freeform.print_plan"), \
             patch("build_loop.modes.freeform.print_final_report"):

            mock_opt.side_effect = PipelineError("Optimization broke tests")
            orch.run("run 2")

        # Run 2's journal must be fresh — no integration failure from run 1
        assert not any(i.kind == IssueKind.INTEGRATION_FAILURE for i in orch.journal.issues)

        # Run 2 must have the optimize failure
        pipeline_errors = [i for i in orch.journal.issues if i.kind == IssueKind.PIPELINE_ERROR]
        assert len(pipeline_errors) == 1, (
            f"Expected 1 PIPELINE_ERROR, got {len(pipeline_errors)}. "
            f"All issues: {[(i.kind.value, i.source.value) for i in orch.journal.issues]}"
        )
        assert pipeline_errors[0].source == IssueSource.OPTIMIZE
