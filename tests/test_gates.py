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
from build_loop.contract import BuildContract
from build_loop.environment import EnvironmentSnapshot
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
