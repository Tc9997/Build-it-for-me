"""Tests for pipeline gating: review rejection, integration failure, smoke testing,
checkpoint/degrade enforcement, plan coverage, mode routing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from build_loop.agents.architect import (
    ArchitectAgent,
    IntegrationFailedError,
    ModuleRejectedError,
    PipelineError,
)
from build_loop.agents.executor import ExecutorAgent
from build_loop.common.pipeline import build_and_review
from build_loop.contract import BuildContract, CapabilityRequirement, CapabilityType
from build_loop.environment import EnvironmentSnapshot
from build_loop.modes import BuildMode
from build_loop.policy import AutonomyMode, PolicyDecision
from build_loop.schemas import (
    BuildArtifact,
    BuildPlan,
    BuildState,
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
            project_name="test", description="test", tech_stack=["python"],
            modules=[ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL)],
            build_order=[["mod_a"]],
        )

    def test_module_rejected_after_max_revisions(self):
        state = BuildState()
        plan = self._make_plan()
        builder = MagicMock()
        reviewer = MagicMock()

        builder.run = MagicMock(return_value=BuildArtifact(
            module_id="mod_a", files={"mod_a.py": "pass"}
        ))
        reviewer.run = MagicMock(return_value=ReviewResult(
            module_id="mod_a", verdict=ReviewVerdict.REVISE, issues=["still broken"],
        ))

        module = plan.modules[0]
        with pytest.raises(ModuleRejectedError) as exc_info:
            build_and_review(module, plan, state, builder, reviewer)

        assert exc_info.value.module_id == "mod_a"
        assert module.status != TaskStatus.APPROVED

    def test_module_approved_on_first_pass(self):
        state = BuildState()
        plan = self._make_plan()
        builder = MagicMock()
        reviewer = MagicMock()

        builder.run = MagicMock(return_value=BuildArtifact(
            module_id="mod_a", files={"mod_a.py": "pass"}
        ))
        reviewer.run = MagicMock(return_value=ReviewResult(
            module_id="mod_a", verdict=ReviewVerdict.APPROVE,
        ))

        artifact, review = build_and_review(plan.modules[0], plan, state, builder, reviewer)
        assert review.verdict == ReviewVerdict.APPROVE


# =========================================================================
# Helper to stub a template_first orchestrator for integration tests
# =========================================================================

def _stub_template_first_agent(output_dir="/tmp/test-build-loop", **kwargs):
    """Create an ArchitectAgent with template_first mode and stub all LLM agents."""
    agent = ArchitectAgent(output_dir=output_dir, mode=BuildMode.TEMPLATE_FIRST, **kwargs)
    orch = agent._orchestrator

    orch.researcher.run = MagicMock(return_value=ResearchReport(
        feasibility="test", recommended_stack=["python"],
    ))
    orch.spec_compiler.run = MagicMock(return_value=BuildContract(
        project_name="test", summary="test",
        goals=["test"], acceptance_criteria=["test"],
        archetype="python_cli",
    ))
    return agent, orch


# =========================================================================
# Integration failure aborts pipeline
# =========================================================================

class TestIntegrationGate:
    """Integration failure must prevent write/setup/test/acceptance phases."""

    def test_integration_failure_stops_pipeline(self):
        agent, orch = _stub_template_first_agent(output_dir="/tmp/test-build-loop-integ")

        with patch("build_loop.modes.template_first.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
            )

            orch.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="stub", name="stub", description="stub", size=TaskSize.SMALL)],
                build_order=[["stub"]],
                goals_covered={"test": ["stub"]},
            ))
            orch.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[], success=False,
                issues=["circular dependency"],
            ))

            # Stub template resolution to avoid real filesystem
            orch._resolve_and_materialize_template = MagicMock()

            # Stub build to avoid real LLM calls
            original_build_all = None
            with patch("build_loop.modes.template_first.build_all") as mock_build:
                with patch("build_loop.modes.template_first.write_project") as mock_write:
                    with patch("build_loop.modes.template_first.setup_environment") as mock_setup:
                        agent.run("test idea")

                        mock_write.assert_not_called()
                        mock_setup.assert_not_called()


# =========================================================================
# Smoke test for long-running services
# =========================================================================

class TestSmokeTestService:
    """A service that stays alive should pass, one that exits immediately should fail."""

    def test_service_that_stays_alive_passes(self, tmp_path):
        executor = ExecutorAgent(str(tmp_path))
        script = tmp_path / "server.py"
        script.write_text("import time; time.sleep(60)")
        result = executor.smoke_test(
            f"{sys.executable} server.py", run_mode="service", timeout=10,
        )
        assert result.success

    def test_service_that_exits_immediately_fails(self, tmp_path):
        executor = ExecutorAgent(str(tmp_path))
        script = tmp_path / "bad_server.py"
        script.write_text("raise SystemExit(1)")
        result = executor.smoke_test(
            f"{sys.executable} bad_server.py", run_mode="service", timeout=10,
        )
        assert not result.success

    def test_batch_mode_checks_exit_code(self, tmp_path):
        executor = ExecutorAgent(str(tmp_path))
        script = tmp_path / "job.py"
        script.write_text("print('done')")
        result = executor.smoke_test(
            f"{sys.executable} job.py", run_mode="batch", timeout=10,
        )
        assert result.success


# =========================================================================
# Executor rejects unsafe commands
# =========================================================================

class TestExecutorSafety:
    def test_rejects_injection(self, tmp_path):
        executor = ExecutorAgent(str(tmp_path))
        result = executor.run_command("pip install foo; rm -rf /")
        assert not result.success
        assert result.exit_code == -2

    def test_accepts_safe_command(self, tmp_path):
        executor = ExecutorAgent(str(tmp_path))
        result = executor.run_command(f"{sys.executable} --version")
        assert result.success


# =========================================================================
# CHECKPOINT mode stops before side-effect phases
# =========================================================================

class TestCheckpointEnforcement:

    def test_checkpoint_stops_without_callback(self):
        agent, orch = _stub_template_first_agent(output_dir="/tmp/test-build-loop-ckpt")

        # Override contract to trigger checkpoint (missing secret)
        orch.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            archetype="python_cli", secrets_required=["MISSING_KEY"],
        ))

        with patch("build_loop.modes.template_first.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True, secrets_missing=["MISSING_KEY"],
            )
            orch._resolve_and_materialize_template = MagicMock()
            orch.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="stub", name="stub", description="stub", size=TaskSize.SMALL)],
                build_order=[["stub"]],
                goals_covered={"test": ["stub"]},
            ))
            orch.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[], success=True,
            ))

            with patch("build_loop.modes.template_first.build_all"):
                with patch("build_loop.modes.template_first.write_project"):
                    with patch("build_loop.modes.template_first.setup_environment") as mock_setup:
                        with patch("build_loop.modes.template_first.test_and_debug_loop"):
                            agent.run("test idea")
                            mock_setup.assert_not_called()

    def test_checkpoint_proceeds_with_confirmation(self):
        confirm_calls = []
        def confirm_yes(phase, reasons):
            confirm_calls.append(phase)
            return True

        agent, orch = _stub_template_first_agent(
            output_dir="/tmp/test-build-loop-ckpt-yes",
            confirm_callback=confirm_yes,
        )
        orch.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            archetype="python_cli", secrets_required=["MISSING_KEY"],
        ))

        with patch("build_loop.modes.template_first.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True, secrets_missing=["MISSING_KEY"],
            )
            orch._resolve_and_materialize_template = MagicMock()
            orch.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="stub", name="stub", description="stub", size=TaskSize.SMALL)],
                build_order=[["stub"]],
                goals_covered={"test": ["stub"]},
            ))
            orch.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[], success=True,
            ))

            with patch("build_loop.modes.template_first.build_all"):
                with patch("build_loop.modes.template_first.write_project"):
                    with patch("build_loop.modes.template_first.setup_environment") as mock_setup:
                        with patch("build_loop.modes.template_first.test_and_debug_loop"):
                            with patch("build_loop.modes.template_first.optimize"):
                                orch._verify = MagicMock()
                                orch._acceptance_check = MagicMock()
                                agent.run("test idea")
                                assert len(confirm_calls) > 0
                                mock_setup.assert_called_once()

    def test_checkpoint_rejected_stops_pipeline(self):
        confirm_calls = []
        def confirm_no(phase, reasons):
            confirm_calls.append(phase)
            return False

        agent, orch = _stub_template_first_agent(
            output_dir="/tmp/test-build-loop-ckpt-no",
            confirm_callback=confirm_no,
        )
        orch.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            archetype="python_cli", secrets_required=["MISSING_KEY"],
        ))

        with patch("build_loop.modes.template_first.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True, secrets_missing=["MISSING_KEY"],
            )
            orch._resolve_and_materialize_template = MagicMock()
            orch.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="stub", name="stub", description="stub", size=TaskSize.SMALL)],
                build_order=[["stub"]],
                goals_covered={"test": ["stub"]},
            ))
            orch.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[], success=True,
            ))

            with patch("build_loop.modes.template_first.build_all"):
                with patch("build_loop.modes.template_first.write_project"):
                    with patch("build_loop.modes.template_first.setup_environment") as mock_setup:
                        agent.run("test idea")
                        assert len(confirm_calls) > 0
                        mock_setup.assert_not_called()


# =========================================================================
# DEGRADE mode skips phases
# =========================================================================

class TestDegradeEnforcement:
    def test_degrade_skips_setup_and_test(self):
        agent, orch = _stub_template_first_agent(output_dir="/tmp/test-build-loop-degrade")
        orch.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            archetype="python_cli",
            capability_requirements=[
                CapabilityRequirement(
                    type=CapabilityType.DOCKER, name="Redis",
                    required=True, affects_phases=["setup", "test", "optimize"],
                ),
            ],
        ))

        with patch("build_loop.modes.template_first.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True, docker_available=False,
            )
            orch._resolve_and_materialize_template = MagicMock()
            orch.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="stub", name="stub", description="stub", size=TaskSize.SMALL)],
                build_order=[["stub"]],
                goals_covered={"test": ["stub"]},
            ))
            orch.integrator.run = MagicMock(return_value=IntegrationResult(
                modules_integrated=[], success=True,
            ))

            with patch("build_loop.modes.template_first.build_all"):
                with patch("build_loop.modes.template_first.write_project") as mock_write:
                    with patch("build_loop.modes.template_first.setup_environment") as mock_setup:
                        with patch("build_loop.modes.template_first.test_and_debug_loop") as mock_test:
                            with patch("build_loop.modes.template_first.optimize") as mock_opt:
                                orch._verify = MagicMock()
                                orch._acceptance_check = MagicMock()
                                agent.run("test idea")

                                mock_write.assert_called_once()
                                mock_setup.assert_not_called()
                                mock_test.assert_not_called()
                                mock_opt.assert_not_called()
                                orch._acceptance_check.assert_called_once()


# =========================================================================
# No verification = INCOMPLETE verdict
# =========================================================================

class TestAcceptanceWithoutVerification:
    def test_no_verification_yields_incomplete(self):
        from build_loop.agents.acceptance import AcceptanceAgent
        from build_loop.schemas import AcceptanceResult, AcceptanceVerdict

        agent = AcceptanceAgent()
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
            verification=None,
        )
        assert result.verdict == AcceptanceVerdict.INCOMPLETE


# =========================================================================
# Plan coverage validation is a hard gate
# =========================================================================

class TestPlanCoverageGate:
    def test_uncovered_goals_stop_pipeline(self):
        agent, orch = _stub_template_first_agent(output_dir="/tmp/test-build-loop-plancov")
        orch.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["goal A", "goal B"],
            acceptance_criteria=["test"],
            archetype="python_cli",
        ))

        with patch("build_loop.modes.template_first.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
            )
            orch._resolve_and_materialize_template = MagicMock()
            # Plan only covers goal A, not goal B
            orch.planner.run = MagicMock(return_value=BuildPlan(
                project_name="test", description="test", tech_stack=["python"],
                modules=[ModuleSpec(id="mod_a", name="A", description="test", size=TaskSize.SMALL)],
                build_order=[["mod_a"]],
                goals_covered={"goal A": ["mod_a"]},
            ))

            with patch("build_loop.modes.template_first.build_all") as mock_build:
                agent.run("test idea")
                mock_build.assert_not_called()


# =========================================================================
# Mode routing
# =========================================================================

class TestModeRouting:
    """Mode selection must route to the correct orchestrator."""

    def test_template_first_is_default(self):
        agent = ArchitectAgent(output_dir="/tmp/test-routing")
        assert agent.mode == BuildMode.TEMPLATE_FIRST

    def test_freeform_mode_creates_freeform_orchestrator(self):
        from build_loop.modes.freeform import FreeformOrchestrator
        agent = ArchitectAgent(output_dir="/tmp/test-routing", mode=BuildMode.FREEFORM)
        assert agent.mode == BuildMode.FREEFORM
        assert isinstance(agent._orchestrator, FreeformOrchestrator)

    def test_template_first_creates_template_orchestrator(self):
        from build_loop.modes.template_first import TemplateFirstOrchestrator
        agent = ArchitectAgent(output_dir="/tmp/test-routing", mode=BuildMode.TEMPLATE_FIRST)
        assert isinstance(agent._orchestrator, TemplateFirstOrchestrator)

    def test_no_silent_fallback(self):
        """template_first must not silently degrade to freeform."""
        from build_loop.modes.freeform import FreeformOrchestrator
        agent = ArchitectAgent(output_dir="/tmp/test-routing", mode=BuildMode.TEMPLATE_FIRST)
        assert not isinstance(agent._orchestrator, FreeformOrchestrator)

    def test_freeform_still_works(self):
        """Freeform orchestrator can be instantiated and has expected attributes."""
        agent = ArchitectAgent(output_dir="/tmp/test-routing", mode=BuildMode.FREEFORM)
        orch = agent._orchestrator
        assert hasattr(orch, "researcher")
        assert hasattr(orch, "planner")
        assert hasattr(orch, "builder")
        assert not hasattr(orch, "spec_compiler")  # Freeform has no contract
        assert not hasattr(orch, "verifier")  # Freeform has no verifier

    def test_freeform_survives_broken_registry(self):
        """Freeform must work even if template registry is broken."""
        # Force the lazy registry to be uninitialized
        import build_loop.templates.registry as reg
        old = reg._REGISTRY
        reg._REGISTRY = None  # Reset to force re-lazy-init

        try:
            # Freeform should import and instantiate without touching registry
            from build_loop.modes.freeform import FreeformOrchestrator
            orch = FreeformOrchestrator(output_dir="/tmp/test-freeform-isolated")
            assert hasattr(orch, "planner")
        finally:
            reg._REGISTRY = old


# =========================================================================
# Template errors flow through controlled error path
# =========================================================================

class TestTemplateErrorHandling:
    """CacheError and MaterializationError must be caught as PipelineError."""

    def test_materialization_error_stops_cleanly(self):
        """MaterializationError should save state and stop, not propagate raw."""
        from build_loop.modes.template_first import TemplateFirstOrchestrator
        from build_loop.templates.materialize import MaterializationError

        orch = TemplateFirstOrchestrator(output_dir="/tmp/test-mat-error")
        orch.researcher.run = MagicMock(return_value=ResearchReport(
            feasibility="test", recommended_stack=[],
        ))
        orch.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            archetype="python_cli",
        ))

        with patch("build_loop.modes.template_first.capture_snapshot") as mock_snap:
            mock_snap.return_value = EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
            )
            # Force materialization to fail
            with patch("build_loop.modes.template_first.materialize_template") as mock_mat:
                mock_mat.side_effect = MaterializationError("ownership.json missing entry")
                with patch("build_loop.modes.template_first.ensure_cached") as mock_cache:
                    mock_cache.return_value = Path("/tmp/fake-cache")

                    # Should NOT raise — should catch and stop cleanly
                    result = orch.run("test idea")

                    # Pipeline stopped, state preserved
                    assert result == orch.output_dir
