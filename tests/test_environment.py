"""Tests for EnvironmentSnapshot capture."""

import os
import sys
from pathlib import Path

import pytest

from build_loop.environment import (
    SCHEMA_VERSION,
    EnvironmentSnapshot,
    capture_snapshot,
)


class TestEnvironmentSnapshot:
    """capture_snapshot must detect real host capabilities."""

    def test_captures_os_info(self):
        snap = capture_snapshot()
        assert snap.os_name  # e.g. "Darwin", "Linux"
        assert snap.arch  # e.g. "arm64", "x86_64"

    def test_captures_python(self):
        snap = capture_snapshot()
        assert snap.python_version
        assert snap.python_path == sys.executable

    def test_detects_git(self):
        snap = capture_snapshot()
        git_tool = next((t for t in snap.tools if t.name == "git"), None)
        assert git_tool is not None
        # Git is almost certainly installed in any dev environment
        assert git_tool.available

    def test_detects_missing_secrets(self):
        snap = capture_snapshot(required_secrets=["DEFINITELY_NOT_SET_XYZ123"])
        assert "DEFINITELY_NOT_SET_XYZ123" in snap.secrets_missing
        assert "DEFINITELY_NOT_SET_XYZ123" not in snap.secrets_present

    def test_detects_present_secrets(self):
        # PATH is always set
        snap = capture_snapshot(required_secrets=["PATH"])
        assert "PATH" in snap.secrets_present
        assert "PATH" not in snap.secrets_missing

    def test_output_dir_writable(self, tmp_path):
        snap = capture_snapshot(output_dir=str(tmp_path))
        assert snap.output_dir_writable

    def test_schema_version(self):
        snap = capture_snapshot()
        assert snap.schema_version == SCHEMA_VERSION

    def test_network_check_returns_bool(self):
        snap = capture_snapshot()
        assert isinstance(snap.network_available, bool)

    def test_required_tools_are_probed(self):
        """Tools from the contract's capability_requirements are discovered."""
        # python3 should be available on any machine running these tests
        snap = capture_snapshot(required_tools=["python3"])
        tool = next((t for t in snap.tools if t.name == "python3"), None)
        assert tool is not None
        assert tool.available

    def test_required_tool_not_installed(self):
        """A tool that doesn't exist is reported as unavailable."""
        snap = capture_snapshot(required_tools=["definitely_not_installed_xyz"])
        tool = next((t for t in snap.tools if t.name == "definitely_not_installed_xyz"), None)
        assert tool is not None
        assert not tool.available

    def test_required_tools_deduplicated_with_builtins(self):
        """Requesting 'git' (a built-in) doesn't produce duplicates."""
        snap = capture_snapshot(required_tools=["git"])
        git_tools = [t for t in snap.tools if t.name == "git"]
        assert len(git_tools) == 1
