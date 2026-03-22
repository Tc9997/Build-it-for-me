"""Tests for the independent verifier.

Proves:
- Signal execution (cli_exit, file_exists, import_check, stdout_contains)
- Tier 1 syntax checking
- Verifier authority: VerificationResult.passed is the truth
- Uncovered behavioral expectations and invariants are reported honestly
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from build_loop.contract import (
    BehavioralExpectation,
    BuildContract,
    CliExitSignal,
    FileExistsSignal,
    HttpProbeSignal,
    ImportCheckSignal,
    Invariant,
    StdoutContainsSignal,
)
from build_loop.verifier import SignalResult, Verifier, VerificationResult


def _make_contract(**overrides) -> BuildContract:
    defaults = dict(
        project_name="test", summary="test",
        goals=["test"], acceptance_criteria=["test"],
        archetype="unsupported",  # Use unsupported to skip tier 3 archetype checks in unit tests
    )
    defaults.update(overrides)
    return BuildContract(**defaults)


class TestTier1Syntax:
    """Tier 1 syntax checks must catch invalid Python."""

    def test_valid_python_passes(self, tmp_path):
        (tmp_path / "good.py").write_text("x = 1\n")
        v = Verifier(str(tmp_path))
        result = v.run(_make_contract())
        syntax_results = [r for r in result.tier1_results if r.signal_type == "syntax"]
        assert all(r.passed for r in syntax_results)

    def test_invalid_python_fails(self, tmp_path):
        (tmp_path / "bad.py").write_text("def f(\n")
        v = Verifier(str(tmp_path))
        result = v.run(_make_contract())
        syntax_results = [r for r in result.tier1_results if r.signal_type == "syntax"]
        assert any(not r.passed for r in syntax_results)
        assert not result.tier1_passed

    def test_skips_venv(self, tmp_path):
        """Files inside .venv should be ignored."""
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "bad.py").write_text("def f(\n")
        (tmp_path / "good.py").write_text("x = 1\n")
        v = Verifier(str(tmp_path))
        result = v.run(_make_contract())
        assert result.tier1_passed


class TestCliExitSignal:
    """cli_exit signals must run the command and check exit code."""

    def test_passing_command(self, tmp_path):
        (tmp_path / "ok.py").write_text("print('ok')\n")
        contract = _make_contract(success_signals=[
            CliExitSignal(description="runs ok", command=sys.executable, args=["ok.py"]),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert result.tier2_results[0].passed

    def test_failing_command(self, tmp_path):
        (tmp_path / "fail.py").write_text("raise SystemExit(1)\n")
        contract = _make_contract(success_signals=[
            CliExitSignal(description="should fail", command=sys.executable, args=["fail.py"]),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert not result.tier2_results[0].passed
        assert not result.tier2_passed

    def test_nonexistent_command(self, tmp_path):
        contract = _make_contract(success_signals=[
            CliExitSignal(description="missing", command="nonexistent_binary_xyz"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert not result.tier2_results[0].passed


class TestShellStyleCommandNormalization:
    """Contract rejects shell-style commands. Verifier normalizes as fallback."""

    def test_contract_rejects_shell_style_cli_exit(self):
        """Shell-style command with empty args fails at contract level."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="single executable"):
            CliExitSignal(
                description="bad",
                command="python ok.py",
                args=[],
            )

    def test_contract_rejects_shell_style_stdout_contains(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="single executable"):
            StdoutContainsSignal(
                description="bad",
                command="python hello.py",
                args=[],
                expect_contains="hello",
            )

    def test_structured_command_still_works(self, tmp_path):
        """Properly structured command+args still works unchanged."""
        (tmp_path / "ok.py").write_text("pass\n")
        contract = _make_contract(success_signals=[
            CliExitSignal(
                description="structured",
                command=sys.executable,
                args=["ok.py"],
                expect_exit=0,
            ),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert result.tier2_results[0].passed

    def test_normalize_argv_splits_shell_style(self):
        result = Verifier._normalize_argv("python -m mypackage.cli version", [])
        assert result == ["python", "-m", "mypackage.cli", "version"]

    def test_normalize_argv_preserves_structured(self):
        result = Verifier._normalize_argv("python", ["-m", "mypackage.cli"])
        assert result == ["python", "-m", "mypackage.cli"]


class TestSafeExecuteRejection:
    """_safe_execute must reject dangerous commands."""

    def test_rejects_shell_interpreter(self, tmp_path):
        """Shell interpreters like /bin/sh must be rejected."""
        v = Verifier(str(tmp_path))
        proc, error = v._safe_execute("/bin/sh", ["-c", "echo pwned"], "shell test")
        assert proc is None
        assert "Rejected" in error

    def test_rejects_python_dash_c(self, tmp_path):
        """python -c must be rejected (arbitrary code execution)."""
        v = Verifier(str(tmp_path))
        proc, error = v._safe_execute("python", ["-c", "import os"], "dash-c test")
        assert proc is None
        assert "Rejected" in error
        assert "-c" in error

    def test_rejects_path_traversal_in_args(self, tmp_path):
        """../../../etc/passwd in args must be rejected."""
        v = Verifier(str(tmp_path))
        proc, error = v._safe_execute("python", ["../../etc/passwd"], "traversal test")
        assert proc is None
        assert "traversal" in error.lower()

    def test_rejects_absolute_executable(self, tmp_path):
        """Absolute paths like /usr/bin/curl must be rejected."""
        v = Verifier(str(tmp_path))
        proc, error = v._safe_execute("/usr/bin/curl", ["http://evil.com"], "abs exec test")
        assert proc is None
        assert "Rejected" in error

    def test_allows_python_dash_m(self, tmp_path):
        """python -m <module> must be allowed."""
        v = Verifier(str(tmp_path))
        proc, error = v._safe_execute("python", ["-m", "json.tool", "--help"], "python -m test")
        # Should not be rejected (may fail if json.tool --help doesn't work, but not rejected)
        assert error is None or "Rejected" not in (error or "")

    def test_allows_venv_executables(self, tmp_path):
        """Executables under .venv/bin/ must be allowed."""
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        fake_tool = venv_bin / "mytool"
        fake_tool.write_text("#!/bin/sh\necho ok")
        fake_tool.chmod(0o755)

        v = Verifier(str(tmp_path))
        proc, error = v._safe_execute(str(fake_tool), ["--version"], "venv tool test")
        # Should not be rejected (may fail to execute, but not rejected by policy)
        assert error is None or "Rejected" not in (error or "")


class TestFileExistsPathSafety:
    """file_exists must reject absolute paths and traversal."""

    def test_rejects_absolute_path(self, tmp_path):
        """file_exists with /etc/passwd must fail, not check the host file."""
        contract = _make_contract(success_signals=[
            FileExistsSignal(description="host file", file_path="/etc/passwd"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        file_results = [r for r in result.tier2_results if r.signal_type == "file_exists"]
        assert len(file_results) == 1
        assert not file_results[0].passed
        assert "rejected" in file_results[0].detail.lower() or "absolute" in file_results[0].detail.lower()

    def test_rejects_dotdot_traversal(self, tmp_path):
        """file_exists with ../../etc/passwd must fail."""
        contract = _make_contract(success_signals=[
            FileExistsSignal(description="traversal", file_path="../../etc/passwd"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        file_results = [r for r in result.tier2_results if r.signal_type == "file_exists"]
        assert not file_results[0].passed
        assert "rejected" in file_results[0].detail.lower() or "traversal" in file_results[0].detail.lower()

    def test_allows_relative_path(self, tmp_path):
        """file_exists with a normal relative path works."""
        (tmp_path / "output.csv").write_text("a,b\n")
        contract = _make_contract(success_signals=[
            FileExistsSignal(description="normal", file_path="output.csv"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        file_results = [r for r in result.tier2_results if r.signal_type == "file_exists"]
        assert file_results[0].passed


class TestImportCheckVenv:
    """Import check should prefer the project venv python."""

    def test_uses_venv_python_when_available(self, tmp_path):
        """If .venv/bin/python exists, it should be used for import checks."""
        # Create a fake venv python that always exits 0
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        fake_python = venv_bin / "python"
        fake_python.write_text("#!/bin/sh\nexit 0\n")
        fake_python.chmod(0o755)

        contract = _make_contract(success_signals=[
            ImportCheckSignal(description="venv import", module_name="anything"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        # The fake python exits 0, so import "passes" — proves venv python was used
        import_results = [r for r in result.tier1_results + result.tier2_results
                          if r.signal_type == "import_check"]
        assert any(r.passed for r in import_results)

    def test_falls_back_to_sys_executable(self, tmp_path):
        """Without a venv, sys.executable is used."""
        # No .venv — should fall back and use sys.executable
        contract = _make_contract(success_signals=[
            ImportCheckSignal(description="stdlib import", module_name="json"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        import_results = [r for r in result.tier1_results + result.tier2_results
                          if r.signal_type == "import_check"]
        assert any(r.passed for r in import_results)


class TestStdoutContainsSignal:
    def test_stdout_match(self, tmp_path):
        (tmp_path / "hello.py").write_text("print('hello world')\n")
        contract = _make_contract(success_signals=[
            StdoutContainsSignal(
                description="outputs hello",
                command=sys.executable, args=["hello.py"],
                expect_contains="hello world",
            ),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert result.tier2_results[0].passed

    def test_stdout_mismatch(self, tmp_path):
        (tmp_path / "hello.py").write_text("print('goodbye')\n")
        contract = _make_contract(success_signals=[
            StdoutContainsSignal(
                description="expects hello",
                command=sys.executable, args=["hello.py"],
                expect_contains="hello world",
            ),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert not result.tier2_results[0].passed


class TestFileExistsSignal:
    def test_file_present(self, tmp_path):
        (tmp_path / "output.csv").write_text("a,b\n1,2\n")
        contract = _make_contract(success_signals=[
            FileExistsSignal(description="output exists", file_path="output.csv"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert result.tier2_results[0].passed

    def test_file_missing(self, tmp_path):
        contract = _make_contract(success_signals=[
            FileExistsSignal(description="output exists", file_path="missing.csv"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert not result.tier2_results[0].passed


class TestImportCheckSignal:
    def test_importable_module(self, tmp_path):
        contract = _make_contract(success_signals=[
            ImportCheckSignal(description="json importable", module_name="json"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        # import_check runs in both tier1 and tier2
        import_results = [r for r in result.tier1_results + result.tier2_results
                          if r.signal_type == "import_check"]
        assert any(r.passed for r in import_results)

    def test_non_importable_module(self, tmp_path):
        contract = _make_contract(success_signals=[
            ImportCheckSignal(description="bogus", module_name="nonexistent_module_xyz"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        import_results = [r for r in result.tier1_results + result.tier2_results
                          if r.signal_type == "import_check"]
        assert any(not r.passed for r in import_results)


class TestVerifierAuthority:
    """VerificationResult.passed is the authoritative verdict."""

    def test_all_pass_means_passed(self, tmp_path):
        (tmp_path / "ok.py").write_text("print('ok')\n")
        contract = _make_contract(success_signals=[
            CliExitSignal(description="runs", command=sys.executable, args=["ok.py"]),
            FileExistsSignal(description="script exists", file_path="ok.py"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert result.passed

    def test_any_fail_means_failed(self, tmp_path):
        (tmp_path / "ok.py").write_text("print('ok')\n")
        contract = _make_contract(success_signals=[
            CliExitSignal(description="runs", command=sys.executable, args=["ok.py"]),
            FileExistsSignal(description="missing", file_path="does_not_exist.csv"),
        ])
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert not result.passed

    def test_syntax_error_fails_tier1(self, tmp_path):
        (tmp_path / "bad.py").write_text("def f(\n")
        v = Verifier(str(tmp_path))
        result = v.run(_make_contract())
        assert not result.tier1_passed
        assert not result.passed


class TestUncoveredItems:
    """Behavioral expectations and invariants must be reported as uncovered."""

    def test_behavioral_expectations_uncovered(self, tmp_path):
        contract = _make_contract(
            behavioral_expectations=[
                BehavioralExpectation(
                    description="parses HTML", given="HTML input", expect="extracted data",
                ),
            ],
        )
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert len(result.uncovered_behavioral) == 1
        assert "parses HTML" in result.uncovered_behavioral[0]

    def test_invariants_uncovered(self, tmp_path):
        contract = _make_contract(
            invariants=[
                Invariant(description="no SQL injection", category="security"),
            ],
        )
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert len(result.uncovered_invariants) == 1
        assert "no SQL injection" in result.uncovered_invariants[0]

    def test_uncovered_does_not_fail_verification(self, tmp_path):
        """Uncovered items are reported but don't cause verification to fail."""
        contract = _make_contract(
            behavioral_expectations=[
                BehavioralExpectation(
                    description="test", given="x", expect="y",
                ),
            ],
            invariants=[Invariant(description="test")],
        )
        v = Verifier(str(tmp_path))
        result = v.run(contract)
        assert result.passed  # No signals to fail


class TestServiceHttpProbe:
    """http_probe on service-mode must start the service under verifier control."""

    def test_service_lifecycle_starts_and_stops(self, tmp_path):
        """Verifier calls _start_service with a probe_url, then _stop_service.

        Uses a long-sleeping process (no socket bind) to prove the lifecycle
        works without requiring loopback bind permissions.
        """
        from unittest.mock import patch, MagicMock

        # Write a script that sleeps (simulates a service that starts)
        script = tmp_path / "fake_service.py"
        script.write_text("import time; time.sleep(300)\n")

        v = Verifier(str(tmp_path))

        # Patch _check_http_probe to return pass (avoids real curl)
        with patch.object(v, "_check_http_probe") as mock_probe:
            mock_probe.return_value = SignalResult(
                signal_type="http_probe", description="mocked", passed=True,
            )
            # Patch _start_service to use a real process but skip readiness polling
            contract = _make_contract(
                run_mode="service",
                success_signals=[
                    HttpProbeSignal(description="health", path="/health"),
                ],
            )
            result = v.run(contract, run_command=f"{sys.executable} fake_service.py")

            # _check_http_probe was called (service lifecycle ran)
            mock_probe.assert_called_once()
            http_results = [r for r in result.tier2_results if r.signal_type == "http_probe"]
            assert len(http_results) == 1
            assert http_results[0].passed

    def test_service_start_failure_reported(self, tmp_path):
        """If the service exits immediately, a service_start failure is recorded."""
        script = tmp_path / "crash.py"
        script.write_text("raise SystemExit(1)\n")

        contract = _make_contract(
            run_mode="service",
            success_signals=[
                HttpProbeSignal(description="will not reach", path="/health"),
            ],
        )
        v = Verifier(str(tmp_path))
        result = v.run(contract, run_command=f"{sys.executable} crash.py")

        service_starts = [r for r in result.tier2_results if r.signal_type == "service_start"]
        assert len(service_starts) == 1
        assert not service_starts[0].passed

    def test_http_probe_without_run_command_does_not_crash(self, tmp_path):
        """http_probe on service-mode without run_command still runs (will fail, not crash)."""
        from unittest.mock import patch

        contract = _make_contract(
            run_mode="service",
            success_signals=[
                HttpProbeSignal(description="will fail", path="/nope"),
            ],
        )
        v = Verifier(str(tmp_path))

        # Patch curl call to avoid network — just return a failed probe
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="000", stderr="", returncode=7,
            )
            result = v.run(contract)  # No run_command

        http_results = [r for r in result.tier2_results if r.signal_type == "http_probe"]
        assert len(http_results) == 1
        assert not http_results[0].passed  # Fails because no server running

    def test_http_probe_batch_mode_no_service_start(self, tmp_path):
        """In batch mode, http_probe should NOT attempt to start a service."""
        from unittest.mock import patch, MagicMock

        contract = _make_contract(
            run_mode="batch",
            success_signals=[
                HttpProbeSignal(description="batch probe", path="/nope"),
            ],
        )
        v = Verifier(str(tmp_path))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="000", stderr="", returncode=7,
            )
            result = v.run(contract, run_command=f"{sys.executable} -c pass")

        # No service_start signal — batch mode doesn't start services
        service_starts = [r for r in result.tier2_results if r.signal_type == "service_start"]
        assert len(service_starts) == 0
