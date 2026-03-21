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

import pytest

from build_loop.contract import (
    BehavioralExpectation,
    BuildContract,
    CliExitSignal,
    FileExistsSignal,
    ImportCheckSignal,
    Invariant,
    StdoutContainsSignal,
)
from build_loop.verifier import Verifier, VerificationResult


def _make_contract(**overrides) -> BuildContract:
    defaults = dict(
        project_name="test", summary="test",
        goals=["test"], acceptance_criteria=["test"],
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
