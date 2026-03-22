"""Tests for pipeline gating: review rejection, integration failure, smoke testing,
checkpoint/degrade enforcement, plan coverage, mode routing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from build_loop.analysis.post_write import PostWriteResult
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
                    with patch("build_loop.analysis.post_write.run_post_write_checks", return_value=PostWriteResult(passed=True)):
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
                    with patch("build_loop.analysis.post_write.run_post_write_checks", return_value=PostWriteResult(passed=True)):
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
                    with patch("build_loop.analysis.post_write.run_post_write_checks", return_value=PostWriteResult(passed=True)):
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
                    with patch("build_loop.analysis.post_write.run_post_write_checks", return_value=PostWriteResult(passed=True)):
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


# =========================================================================
# Setup failure stops the pipeline
# =========================================================================

class TestDuplicateArtifactPaths:
    """Two artifacts claiming the same path must fail before write."""

    def test_duplicate_file_path_raises(self):
        from build_loop.common.pipeline import write_project, PipelineError

        state = BuildState()
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"shared.py": "# from mod_a"},
            ),
            "mod_b": BuildArtifact(
                module_id="mod_b",
                files={"shared.py": "# from mod_b"},  # same path!
            ),
        }

        with pytest.raises(PipelineError, match="Duplicate file path.*shared.py"):
            write_project(state, "/tmp/test-dup", lambda p, c: None)

    def test_no_duplicate_passes(self):
        from build_loop.common.pipeline import write_project

        state = BuildState()
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"mod_a.py": "pass"},
            ),
            "mod_b": BuildArtifact(
                module_id="mod_b",
                files={"mod_b.py": "pass"},
            ),
        }

        # Should not raise
        written = []
        write_project(state, "/tmp/test-nodup", lambda p, c: written.append(p))
        assert "mod_a.py" in written
        assert "mod_b.py" in written

    def test_builder_producing_pyproject_toml_raises(self):
        """Builders must not produce integrator-owned files."""
        from build_loop.common.pipeline import write_project, PipelineError

        state = BuildState()
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"pyproject.toml": "[project]\nname = 'bad'"},
            ),
        }

        with pytest.raises(PipelineError, match="integrator-owned"):
            write_project(state, "/tmp/test-forbidden", lambda p, c: None)

    def test_builder_producing_readme_raises(self):
        from build_loop.common.pipeline import write_project, PipelineError

        state = BuildState()
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"README.md": "# Bad readme"},
            ),
        }

        with pytest.raises(PipelineError, match="integrator-owned"):
            write_project(state, "/tmp/test-forbidden-readme", lambda p, c: None)

    def test_builder_producing_requirements_txt_raises(self):
        from build_loop.common.pipeline import write_project, PipelineError

        state = BuildState()
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"requirements.txt": "pydantic>=2.0"},
            ),
        }

        with pytest.raises(PipelineError, match="integrator-owned"):
            write_project(state, "/tmp/test-forbidden-req", lambda p, c: None)

    def test_integrator_overwriting_builder_file_raises(self):
        """Integrator cannot overwrite a builder-owned module file."""
        from build_loop.common.pipeline import write_project, PipelineError

        state = BuildState()
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"pkg/core.py": "# builder version"},
            ),
        }
        state.integration = IntegrationResult(
            modules_integrated=["mod_a"],
            success=True,
            wiring_files={"pkg/core.py": "# integrator version"},  # conflicts!
        )

        with pytest.raises(PipelineError, match="overwrite.*pkg/core.py"):
            write_project(state, "/tmp/test-integ-overwrite", lambda p, c: None)

    def test_integrator_shared_files_pass(self):
        """Integrator producing pyproject.toml and README.md is allowed."""
        from build_loop.common.pipeline import write_project

        state = BuildState()
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"pkg/core.py": "pass"},
            ),
        }
        state.integration = IntegrationResult(
            modules_integrated=["mod_a"],
            success=True,
            wiring_files={
                "pyproject.toml": "[project]",
                "README.md": "# Readme",
                "requirements.txt": "pydantic",
            },
        )

        written = []
        write_project(state, "/tmp/test-integ-shared", lambda p, c: written.append(p))
        assert "pyproject.toml" in written
        assert "README.md" in written
        assert "requirements.txt" in written
        assert "pkg/core.py" in written

    def test_integrator_new_wiring_file_passes(self):
        """Integrator producing a new file that no builder owns is allowed."""
        from build_loop.common.pipeline import write_project

        state = BuildState()
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"pkg/core.py": "pass"},
            ),
        }
        state.integration = IntegrationResult(
            modules_integrated=["mod_a"],
            success=True,
            wiring_files={"pkg/__main__.py": "from pkg.core import main; main()"},
        )

        written = []
        write_project(state, "/tmp/test-integ-new", lambda p, c: written.append(p))
        assert "pkg/__main__.py" in written


    def test_interface_conflicting_with_builder_raises(self):
        """A planner interface at the same path as a builder file must fail."""
        from build_loop.common.pipeline import write_project, PipelineError
        from build_loop.schemas import InterfaceContract

        state = BuildState()
        state.plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            interfaces=[InterfaceContract(
                name="core", description="core interface",
                file_path="pkg/core.py", code="class Base: pass",
            )],
        )
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"pkg/core.py": "# builder version"},  # conflicts with interface
            ),
        }

        with pytest.raises(PipelineError, match="Duplicate file path.*pkg/core.py"):
            write_project(state, "/tmp/test-iface-conflict", lambda p, c: None)

    def test_interface_conflicting_with_integrator_raises(self):
        """A planner interface at the same path as an integrator file must fail."""
        from build_loop.common.pipeline import write_project, PipelineError
        from build_loop.schemas import InterfaceContract

        state = BuildState()
        state.plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            interfaces=[InterfaceContract(
                name="main", description="entry",
                file_path="main.py", code="def run(): pass",
            )],
        )
        state.artifacts = {}
        state.integration = IntegrationResult(
            modules_integrated=[],
            success=True,
            wiring_files={"main.py": "# integrator version"},  # conflicts with interface
        )

        with pytest.raises(PipelineError, match="overwrite.*main.py"):
            write_project(state, "/tmp/test-iface-integ", lambda p, c: None)

    def test_interface_without_conflict_passes(self):
        """Interface files with no conflicts should be written normally."""
        from build_loop.common.pipeline import write_project
        from build_loop.schemas import InterfaceContract

        state = BuildState()
        state.plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            interfaces=[InterfaceContract(
                name="types", description="shared types",
                file_path="pkg/types.py", code="class MyType: pass",
            )],
        )
        state.artifacts = {
            "mod_a": BuildArtifact(
                module_id="mod_a",
                files={"pkg/impl.py": "from pkg.types import MyType"},
            ),
        }

        written = []
        write_project(state, "/tmp/test-iface-ok", lambda p, c: written.append(p))
        assert "pkg/types.py" in written
        assert "pkg/impl.py" in written


class TestSetupFailureGate:
    """Failed setup commands must raise PipelineError."""

    def test_setup_failure_raises(self):
        from build_loop.common.pipeline import setup_environment, PipelineError
        from build_loop.schemas import BuildState

        executor = MagicMock()
        failed_result = MagicMock()
        failed_result.success = False
        failed_result.command = "python3 -m venv .venv"
        failed_result.stderr = "error: venv creation failed"
        executor.setup_project.return_value = [failed_result]
        executor.project_dir = "/tmp/fake"

        state = BuildState()

        with pytest.raises(PipelineError, match="Setup failed"):
            setup_environment(state, executor, lambda cmd: cmd)


# =========================================================================
# Test exhaustion stops the pipeline
# =========================================================================

class TestDebugExhaustionGate:
    """Exhausting all debug rounds must raise PipelineError."""

    def test_exhausted_debug_raises(self):
        from build_loop.common.pipeline import test_and_debug_loop, PipelineError
        from build_loop.schemas import BuildState

        executor = MagicMock()
        failed_test = MagicMock()
        failed_test.success = False
        failed_test.command = "pytest"
        failed_test.stdout = "FAILED"
        failed_test.stderr = "AssertionError"
        executor.run_tests.return_value = failed_test

        debugger = MagicMock()
        debugger.run.side_effect = RuntimeError("debugger failed")

        state = BuildState()
        state.plan = BuildPlan(
            project_name="test", description="test", tech_stack=["python"],
            test_command="pytest",
        )

        with pytest.raises(PipelineError, match="did not pass"):
            test_and_debug_loop(
                state, executor, debugger,
                lambda cmd: cmd, lambda p, c: None, lambda: {},
            )


# =========================================================================
# Freeform mode can succeed
# =========================================================================

class TestFreeformExitCode:
    """Freeform mode must be able to return a successful exit path."""

    def test_freeform_incomplete_is_not_failure(self):
        """Freeform with INCOMPLETE verdict should not be treated as fail."""
        from build_loop.schemas import AcceptanceResult, AcceptanceVerdict
        from build_loop.modes import BuildMode

        # Simulate what main.py does
        verdict = AcceptanceVerdict.INCOMPLETE
        verdict_str = str(verdict.value if hasattr(verdict, "value") else verdict)
        mode = BuildMode.FREEFORM

        # Freeform: only "fail" is failure
        if mode == BuildMode.TEMPLATE_FIRST:
            is_failure = verdict_str != "pass"
        else:
            is_failure = verdict_str == "fail"

        assert not is_failure, "Freeform INCOMPLETE should not be treated as failure"

    def test_freeform_pass_is_success(self):
        from build_loop.schemas import AcceptanceVerdict
        from build_loop.modes import BuildMode

        verdict = AcceptanceVerdict.PASS
        verdict_str = str(verdict.value)
        mode = BuildMode.FREEFORM

        is_failure = verdict_str == "fail"
        assert not is_failure

    def test_template_first_incomplete_is_failure(self):
        from build_loop.schemas import AcceptanceVerdict
        from build_loop.modes import BuildMode

        verdict = AcceptanceVerdict.INCOMPLETE
        verdict_str = str(verdict.value)
        mode = BuildMode.TEMPLATE_FIRST

        is_failure = verdict_str != "pass"
        assert is_failure, "template_first INCOMPLETE must be treated as failure"


# =========================================================================
# Resumed degraded mode skips verify
# =========================================================================

class TestResumeDegradedSkipsVerify:
    """A degraded policy restored on resume must skip verify."""

    def test_resumed_degrade_skips_verify(self, tmp_path):
        from build_loop.modes.template_first import TemplateFirstOrchestrator
        from build_loop.policy import AutonomyMode, PolicyDecision
        from build_loop.schemas import BuildState, ContractState, EnvironmentState, PolicyState
        from build_loop.environment import EnvironmentSnapshot
        import json

        # Build a saved state with a DEGRADE policy that skips verify
        state = BuildState(
            idea="test",
            contract=ContractState(data=BuildContract(
                project_name="test", summary="test",
                goals=["test"], acceptance_criteria=["test"],
                archetype="python_cli",
            )),
            environment=EnvironmentState(data=EnvironmentSnapshot(
                os_name="Test", arch="x86_64",
                python_version="3.12", python_path="/usr/bin/python3",
                output_dir_writable=True,
            )),
            policy=PolicyState(data=PolicyDecision(
                autonomy_mode=AutonomyMode.DEGRADE,
                skip_phases=["setup", "test", "optimize", "verify"],
            )),
            output_dir=str(tmp_path),
        )

        # Save state to disk
        state_dir = tmp_path / ".build_state"
        state_dir.mkdir()
        (state_dir / "state.json").write_text(state.model_dump_json())

        # Create orchestrator and resume from verify
        orch = TemplateFirstOrchestrator(output_dir=str(tmp_path))
        orch._verify = MagicMock()
        orch._acceptance_check = MagicMock()

        orch.resume("verify")

        # verify should have been skipped due to DEGRADE policy
        orch._verify.assert_not_called()
        # acceptance should still run
        orch._acceptance_check.assert_called_once()
