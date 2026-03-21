"""Tests for pipeline gating: review rejection, integration failure, smoke testing."""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from build_loop.agents.architect import (
    ArchitectAgent,
    IntegrationFailedError,
    ModuleRejectedError,
)
from build_loop.agents.executor import ExecutorAgent
from build_loop.contract import BuildContract, CapabilityRequirement, CapabilityType
from build_loop.environment import EnvironmentSnapshot
from build_loop.agents.architect import PipelineError
from build_loop.policy import AutonomyMode, PolicyDecision
from build_loop.schemas import (
    BuildArtifact,
    BuildPlan,
    ExecResult,
    IntegrationResult,
    ModuleSpec,
    ResearchReport,
    ReviewResult,
    ReviewVerdict,
    TaskSize,
    TaskStatus,
)


# =========================================================================
# Review rejection blocks approval
# =========================================================================

class TestReviewGate:
    """A module that gets 'revise' three times must never reach APPROVED."""

    def _make_plan(self) -> BuildPlan:
        return BuildPlan(
            project_name="test",
            description="test",
            tech_stack=["python"],
            modules=[ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL)],
            build_order=[["mod_a"]],
        )

    def test_module_rejected_after_max_revisions(self):
        agent = ArchitectAgent(output_dir="/tmp/test-build-loop-gate")
        plan = self._make_plan()

        # Mock builder to always return an artifact
        agent.builder.run = MagicMock(return_value=BuildArtifact(
            module_id="mod_a", files={"mod_a.py": "pass"}
        ))

        # Mock reviewer to always return REVISE
        agent.reviewer.run = MagicMock(return_value=ReviewResult(
            module_id="mod_a",
            verdict=ReviewVerdict.REVISE,
            issues=["still broken"],
        ))

        module = plan.modules[0]
        with pytest.raises(ModuleRejectedError) as exc_info:
            agent._build_and_review(module, plan)

        assert exc_info.value.module_id == "mod_a"
        assert module.status != TaskStatus.APPROVED

    def test_module_approved_on_first_pass(self):
        agent = ArchitectAgent(output_dir="/tmp/test-build-loop-gate")
        plan = self._make_plan()

        agent.builder.run = MagicMock(return_value=BuildArtifact(
            module_id="mod_a", files={"mod_a.py": "pass"}
        ))
        agent.reviewer.run = MagicMock(return_value=ReviewResult(
            module_id="mod_a",
            verdict=ReviewVerdict.APPROVE,
        ))

        artifact, review = agent._build_and_review(plan.modules[0], plan)
        assert review.verdict == ReviewVerdict.APPROVE


# =========================================================================
# Integration failure aborts pipeline
# =========================================================================

class TestIntegrationGate:
    """Integration failure must prevent write/setup/test/acceptance phases."""

    def test_integration_failure_stops_pipeline(self):
        agent = ArchitectAgent(output_dir="/tmp/test-build-loop-integ")

        # Stub research
        agent.researcher.run = MagicMock(return_value=ResearchReport(
            feasibility="test", recommended_stack=["python"],
        ))

        # Stub contract / environment / policy
        agent.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
        ))
        # Patch capture_snapshot to avoid real OS inspection
        with patch("build_loop.agents.architect.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
            )

            agent.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[], build_order=[],
            ))

            # Return a failed integration
            agent.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[],
                success=False,
                issues=["circular dependency between A and B"],
            ))

            # These should never be called
            agent._write_project = MagicMock()
            agent._setup_environment = MagicMock()
            agent._test_and_debug_loop = MagicMock()
            agent._optimize = MagicMock()
            agent._acceptance_check = MagicMock()

            agent.run("test idea")

            agent._write_project.assert_not_called()
            agent._setup_environment.assert_not_called()
            agent._test_and_debug_loop.assert_not_called()
            agent._optimize.assert_not_called()
            agent._acceptance_check.assert_not_called()


# =========================================================================
# Smoke test for long-running services
# =========================================================================

class TestSmokeTestService:
    """A service that stays alive should pass, one that exits immediately should fail."""

    def test_service_that_stays_alive_passes(self, tmp_path):
        """A long-running process that's still alive after the probe window is healthy."""
        executor = ExecutorAgent(str(tmp_path))

        # Write a simple script that sleeps (simulates a server)
        script = tmp_path / "server.py"
        script.write_text("import time; time.sleep(60)")

        result = executor.smoke_test(
            f"{sys.executable} server.py",
            run_mode="service",
            timeout=10,
        )
        assert result.success, f"Service smoke test failed: {result.stderr}"

    def test_service_that_exits_immediately_fails(self, tmp_path):
        """A service that crashes on startup should fail the smoke test."""
        executor = ExecutorAgent(str(tmp_path))

        script = tmp_path / "bad_server.py"
        script.write_text("raise SystemExit(1)")

        result = executor.smoke_test(
            f"{sys.executable} bad_server.py",
            run_mode="service",
            timeout=10,
        )
        assert not result.success

    def test_batch_mode_checks_exit_code(self, tmp_path):
        """Batch mode should use exit code, not liveness."""
        executor = ExecutorAgent(str(tmp_path))

        script = tmp_path / "job.py"
        script.write_text("print('done')")

        result = executor.smoke_test(
            f"{sys.executable} job.py",
            run_mode="batch",
            timeout=10,
        )
        assert result.success


# =========================================================================
# Executor rejects unsafe commands
# =========================================================================

class TestExecutorSafety:
    """Executor must reject commands with shell metacharacters."""

    def test_rejects_injection(self, tmp_path):
        executor = ExecutorAgent(str(tmp_path))
        result = executor.run_command("pip install foo; rm -rf /")
        assert not result.success
        assert result.exit_code == -2
        assert "rejected" in result.stderr.lower()

    def test_accepts_safe_command(self, tmp_path):
        executor = ExecutorAgent(str(tmp_path))
        result = executor.run_command(f"{sys.executable} --version")
        assert result.success


# =========================================================================
# Helper to build a fully-stubbed architect for policy tests
# =========================================================================

def _make_stubbed_architect(policy_decision, output_dir="/tmp/test-build-loop-policy"):
    """Create an ArchitectAgent with all LLM agents stubbed, using the given policy."""
    agent = ArchitectAgent(output_dir=output_dir)

    agent.researcher.run = MagicMock(return_value=ResearchReport(
        feasibility="test", recommended_stack=["python"],
    ))
    agent.spec_compiler.run = MagicMock(return_value=BuildContract(
        project_name="test", summary="test",
        goals=["test"], acceptance_criteria=["test"],
    ))
    agent.planner.run = MagicMock(return_value=BuildPlan(
        project_name="test", description="test", tech_stack=["python"],
        modules=[], build_order=[],
    ))
    agent.integrator.run = MagicMock(return_value=IntegrationResult(
        modules_integrated=[], success=True,
    ))

    # Inject policy decision directly
    agent.policy_decision = policy_decision
    # Override evaluate_policy to return our decision
    return agent


# =========================================================================
# CHECKPOINT mode stops before side-effect phases
# =========================================================================

class TestCheckpointEnforcement:
    """CHECKPOINT must stop before side-effect phases unless confirmed."""

    def test_checkpoint_stops_without_callback(self):
        """No confirm callback → pipeline stops at checkpoint (fail-closed)."""
        agent = ArchitectAgent(output_dir="/tmp/test-build-loop-ckpt")

        agent.researcher.run = MagicMock(return_value=ResearchReport(
            feasibility="test", recommended_stack=["python"],
        ))
        agent.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            secrets_required=["MISSING_KEY"],
        ))

        with patch("build_loop.agents.architect.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
                secrets_missing=["MISSING_KEY"],
            )

            # No confirm callback — should stop at checkpoint
            agent._confirm = None

            agent.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[], build_order=[],
            ))
            agent.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[], success=True,
            ))

            agent._build_all = MagicMock()
            agent._write_project = MagicMock()
            # These should never be called (checkpoint blocks before setup)
            agent._setup_environment = MagicMock()
            agent._test_and_debug_loop = MagicMock()

            agent.run("test idea")

            agent._setup_environment.assert_not_called()
            agent._test_and_debug_loop.assert_not_called()

    def test_checkpoint_proceeds_with_confirmation(self):
        """With a confirm callback that returns True, pipeline proceeds past checkpoints."""
        confirm_calls = []

        def confirm_yes(phase, reasons):
            confirm_calls.append(phase)
            return True

        agent = ArchitectAgent(
            output_dir="/tmp/test-build-loop-ckpt-yes",
            confirm_callback=confirm_yes,
        )

        agent.researcher.run = MagicMock(return_value=ResearchReport(
            feasibility="test", recommended_stack=["python"],
        ))
        agent.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            secrets_required=["MISSING_KEY"],
        ))

        with patch("build_loop.agents.architect.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
                secrets_missing=["MISSING_KEY"],
            )

            agent.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="stub", name="stub", description="stub", size=TaskSize.SMALL)],
                build_order=[["stub"]],
                goals_covered={"test": ["stub"]},
            ))
            agent.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[], success=True,
            ))

            # Stub build and side-effect phases
            agent._build_all = MagicMock()
            agent._write_project = MagicMock()
            agent._setup_environment = MagicMock()
            agent._test_and_debug_loop = MagicMock()
            agent._optimize = MagicMock()
            agent._acceptance_check = MagicMock()

            agent.run("test idea")

            # Confirm callback was called for the checkpoint phases
            assert len(confirm_calls) > 0
            # Side-effect phases DID execute because confirmation was given
            agent._setup_environment.assert_called_once()

    def test_checkpoint_rejected_stops_pipeline(self):
        """Confirm callback returns False → pipeline stops at checkpoint, not earlier."""
        confirm_calls = []

        def confirm_no(phase, reasons):
            confirm_calls.append(phase)
            return False

        agent = ArchitectAgent(
            output_dir="/tmp/test-build-loop-ckpt-no",
            confirm_callback=confirm_no,
        )

        agent.researcher.run = MagicMock(return_value=ResearchReport(
            feasibility="test", recommended_stack=["python"],
        ))
        agent.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            secrets_required=["MISSING_KEY"],
        ))

        with patch("build_loop.agents.architect.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
                secrets_missing=["MISSING_KEY"],
            )

            agent.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="stub", name="stub", description="stub", size=TaskSize.SMALL)],
                build_order=[["stub"]],
                goals_covered={"test": ["stub"]},
            ))

            agent.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[], success=True,
            ))

            agent._build_all = MagicMock()
            agent._write_project = MagicMock()
            agent._setup_environment = MagicMock()

            agent.run("test idea")

            # Confirm callback was actually called (not short-circuited by plan gate)
            assert len(confirm_calls) > 0
            # Setup should not have been called — checkpoint rejected
            agent._setup_environment.assert_not_called()


# =========================================================================
# DEGRADE mode skips phases
# =========================================================================

class TestDegradeEnforcement:
    """DEGRADE must skip phases listed in skip_phases."""

    def test_degrade_skips_setup_and_test(self):
        """When policy says degrade with skip_phases, those phases don't execute."""
        agent = ArchitectAgent(output_dir="/tmp/test-build-loop-degrade")

        agent.researcher.run = MagicMock(return_value=ResearchReport(
            feasibility="test", recommended_stack=["python"],
        ))
        agent.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            capability_requirements=[
                CapabilityRequirement(
                    type=CapabilityType.DOCKER, name="Redis",
                    required=True, affects_phases=["setup", "test", "optimize"],
                ),
            ],
        ))

        with patch("build_loop.agents.architect.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
                docker_available=False,
            )

            agent.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="stub", name="stub", description="stub", size=TaskSize.SMALL)],
                build_order=[["stub"]],
                goals_covered={"test": ["stub"]},
            ))
            agent.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[], success=True,
            ))

            # Stub build to avoid "no modules approved" error
            agent._build_all = MagicMock()
            agent._write_project = MagicMock()
            agent._setup_environment = MagicMock()
            agent._test_and_debug_loop = MagicMock()
            agent._optimize = MagicMock()
            agent._acceptance_check = MagicMock()

            agent.run("test idea")

            # Write still happens — it's not a skippable phase
            agent._write_project.assert_called_once()
            # Setup, test, and optimize are skipped by degrade policy
            agent._setup_environment.assert_not_called()
            agent._test_and_debug_loop.assert_not_called()
            agent._optimize.assert_not_called()
            # Acceptance still happens
            agent._acceptance_check.assert_called_once()


# =========================================================================
# No verification = INCOMPLETE verdict
# =========================================================================

class TestAcceptanceWithoutVerification:
    """Acceptance without verification must yield INCOMPLETE, not PASS."""

    def test_no_verification_yields_incomplete(self):
        from build_loop.agents.acceptance import AcceptanceAgent
        from build_loop.schemas import AcceptanceResult, AcceptanceVerdict

        agent = AcceptanceAgent()

        # Mock the LLM call to return a "pass" verdict
        agent.call_json = MagicMock(return_value={
            "verdict": "pass",
            "criteria_checked": ["looks good"],
            "criteria_passed": ["looks good"],
            "criteria_failed": [],
            "notes": "LLM says pass",
        })

        result = agent.run(
            idea="test",
            plan=BuildPlan(project_name="test", description="test", tech_stack=["python"]),
            project_files={},
            verification=None,  # No verification
        )

        # Even though LLM said pass, verdict must be INCOMPLETE
        assert result.verdict == AcceptanceVerdict.INCOMPLETE
        assert "skipped" in result.notes.lower() or "cannot confirm" in result.notes.lower()


# =========================================================================
# Plan coverage validation is a hard gate
# =========================================================================

class TestPlanCoverageGate:
    """Invalid plan coverage must stop the pipeline."""

    def test_uncovered_goals_stop_pipeline(self):
        """A plan missing contract goals should not reach build phase."""
        from build_loop.schemas import ModuleSpec, TaskSize

        agent = ArchitectAgent(output_dir="/tmp/test-build-loop-plancov")

        agent.researcher.run = MagicMock(return_value=ResearchReport(
            feasibility="test", recommended_stack=["python"],
        ))
        agent.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["goal A", "goal B"],
            acceptance_criteria=["test"],
        ))

        with patch("build_loop.agents.architect.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
            )

            # Planner returns a plan that only covers goal A, not goal B
            agent.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL)],
                build_order=[["mod_a"]],
                goals_covered={"goal A": ["mod_a"]},
                # goal B is missing!
            ))

            agent._build_all = MagicMock()
            agent.run("test idea")

            # Build should never have been called — plan validation gate stopped it
            agent._build_all.assert_not_called()
