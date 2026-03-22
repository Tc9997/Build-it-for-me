"""Tests for engine, routing, and runtime abstractions.

Proves:
- RouteDecision captures mode selection with promise level
- is_success matches existing behavior for both modes
- Router produces correct decisions for both modes
- RuntimeServices wraps filesystem ops correctly
- ArchitectAgent.is_success() works through the abstraction
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from build_loop.engine import (
    BuildEngine,
    EngineCapabilities,
    PromiseLevel,
    RouteDecision,
)
from build_loop.modes import BuildMode
from build_loop.routing import (
    FREEFORM_CAPABILITIES,
    TEMPLATE_FIRST_CAPABILITIES,
    is_success,
    route,
)
from build_loop.runtime import ProjectIO, RuntimeServices
from build_loop.schemas import BuildState


# =========================================================================
# Engine abstraction
# =========================================================================

class TestEngineCapabilities:
    def test_template_first_is_verified(self):
        assert TEMPLATE_FIRST_CAPABILITIES.promise_level == PromiseLevel.VERIFIED
        assert TEMPLATE_FIRST_CAPABILITIES.supports_resume
        assert TEMPLATE_FIRST_CAPABILITIES.supports_verify
        assert "python_cli" in TEMPLATE_FIRST_CAPABILITIES.supported_archetypes

    def test_freeform_is_best_effort(self):
        assert FREEFORM_CAPABILITIES.promise_level == PromiseLevel.BEST_EFFORT
        assert not FREEFORM_CAPABILITIES.supports_resume
        assert not FREEFORM_CAPABILITIES.supports_verify
        assert FREEFORM_CAPABILITIES.supported_archetypes == []


# =========================================================================
# Route decisions
# =========================================================================

class TestRouteDecision:
    def test_template_first_route(self):
        decision = route(BuildMode.TEMPLATE_FIRST)
        assert decision.engine_name == "template_first"
        assert decision.promise_level == PromiseLevel.VERIFIED
        assert decision.confidence == 1.0
        assert decision.mode_value == "template_first"

    def test_freeform_route(self):
        decision = route(BuildMode.FREEFORM)
        assert decision.engine_name == "freeform"
        assert decision.promise_level == PromiseLevel.BEST_EFFORT
        assert decision.mode_value == "freeform"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            route("nonexistent")


class TestIsSuccess:
    """is_success must match existing behavior exactly."""

    def test_verified_pass_is_success(self):
        d = RouteDecision(engine_name="t", promise_level=PromiseLevel.VERIFIED, mode_value="template_first")
        assert is_success(d, "pass")

    def test_verified_fail_is_not_success(self):
        d = RouteDecision(engine_name="t", promise_level=PromiseLevel.VERIFIED, mode_value="template_first")
        assert not is_success(d, "fail")

    def test_verified_incomplete_is_not_success(self):
        d = RouteDecision(engine_name="t", promise_level=PromiseLevel.VERIFIED, mode_value="template_first")
        assert not is_success(d, "incomplete")

    def test_best_effort_pass_is_success(self):
        d = RouteDecision(engine_name="f", promise_level=PromiseLevel.BEST_EFFORT, mode_value="freeform")
        assert is_success(d, "pass")

    def test_best_effort_incomplete_is_success(self):
        d = RouteDecision(engine_name="f", promise_level=PromiseLevel.BEST_EFFORT, mode_value="freeform")
        assert is_success(d, "incomplete")

    def test_best_effort_fail_is_not_success(self):
        d = RouteDecision(engine_name="f", promise_level=PromiseLevel.BEST_EFFORT, mode_value="freeform")
        assert not is_success(d, "fail")


# =========================================================================
# Runtime services
# =========================================================================

class TestProjectIO:
    def test_safe_write_creates_file(self, tmp_path):
        io = ProjectIO(str(tmp_path))
        io.safe_write("src/main.py", "print('hello')")
        assert (tmp_path / "src" / "main.py").read_text() == "print('hello')"

    def test_safe_write_rejects_traversal(self, tmp_path):
        from build_loop.safety import PathTraversalError
        io = ProjectIO(str(tmp_path))
        with pytest.raises(PathTraversalError):
            io.safe_write("../../etc/passwd", "evil")

    def test_read_project_files(self, tmp_path):
        io = ProjectIO(str(tmp_path))
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.txt").write_text("hello")
        files = io.read_project_files()
        assert "a.py" in files
        assert "b.txt" in files

    def test_save_and_load_state(self, tmp_path):
        io = ProjectIO(str(tmp_path))
        state = BuildState(idea="test project", output_dir=str(tmp_path))
        io.save_state(state)
        loaded = io.load_state()
        assert loaded is not None
        assert loaded.idea == "test project"

    def test_load_state_missing(self, tmp_path):
        io = ProjectIO(str(tmp_path))
        assert io.load_state() is None


class TestRuntimeServices:
    def test_venv_python_absent(self, tmp_path):
        rt = RuntimeServices(str(tmp_path))
        assert rt.venv_python is None

    def test_venv_python_present(self, tmp_path):
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("#!/bin/sh")
        (venv / "python").chmod(0o755)
        rt = RuntimeServices(str(tmp_path))
        assert rt.venv_python is not None
        assert "python" in rt.venv_python


# =========================================================================
# Router uses abstractions
# =========================================================================

class TestRouterAbstraction:
    """ArchitectAgent uses RouteDecision and is_success()."""

    def test_architect_has_route_decision(self):
        from build_loop.agents.architect import ArchitectAgent
        agent = ArchitectAgent(output_dir="/tmp/test-abstraction")
        assert agent.decision is not None
        assert agent.decision.engine_name == "template_first"
        assert agent.decision.promise_level == PromiseLevel.VERIFIED

    def test_architect_freeform_decision(self):
        from build_loop.agents.architect import ArchitectAgent
        agent = ArchitectAgent(output_dir="/tmp/test-abstraction", mode=BuildMode.FREEFORM)
        assert agent.decision.engine_name == "freeform"
        assert agent.decision.promise_level == PromiseLevel.BEST_EFFORT

    def test_is_success_no_acceptance(self):
        from build_loop.agents.architect import ArchitectAgent
        agent = ArchitectAgent(output_dir="/tmp/test-abstraction")
        assert not agent.is_success()

    def test_selection_behavior_unchanged(self):
        """template_first and freeform still route to the correct orchestrators."""
        from build_loop.agents.architect import ArchitectAgent
        from build_loop.modes.freeform import FreeformOrchestrator

        tf = ArchitectAgent(output_dir="/tmp/test-tf", mode=BuildMode.TEMPLATE_FIRST)
        ff = ArchitectAgent(output_dir="/tmp/test-ff", mode=BuildMode.FREEFORM)

        assert not isinstance(tf._orchestrator, FreeformOrchestrator)
        assert isinstance(ff._orchestrator, FreeformOrchestrator)
