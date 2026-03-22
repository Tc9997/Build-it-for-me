"""Derive success signals deterministically from contract archetype + goals.

Replaces LLM-generated signals with correctly structured ones.
The spec compiler still produces goals, constraints, behavioral_expectations,
and invariants — just not the machine-checkable signals.
"""

from __future__ import annotations

from build_loop.contract import (
    BuildContract,
    CliExitSignal,
    FileExistsSignal,
    ImportCheckSignal,
    StdoutContainsSignal,
)


def derive_signals(contract: BuildContract) -> list:
    """Derive success signals from contract archetype and goals.

    Returns deterministic signals based on the archetype pattern
    and contract field values. Always correctly structured.
    """
    signals: list = []
    pkg = contract.project_name.replace("-", "_")

    # Universal: package importable
    signals.append(ImportCheckSignal(
        description=f"Package '{pkg}' is importable",
        module_name=pkg,
    ))

    # Universal: pyproject.toml exists
    signals.append(FileExistsSignal(
        description="pyproject.toml exists",
        file_path="pyproject.toml",
    ))

    if contract.archetype == "python_cli":
        signals.extend(_python_cli_signals(contract, pkg))
    elif contract.archetype == "fastapi_service":
        signals.extend(_fastapi_service_signals(contract, pkg))

    # PEP 561 if goals mention type/typed
    if any("type" in g.lower() or "typed" in g.lower() or "py.typed" in g.lower()
           for g in contract.goals):
        signals.append(FileExistsSignal(
            description="PEP 561 py.typed marker exists",
            file_path=f"{pkg}/py.typed",
        ))

    return signals


def _python_cli_signals(contract: BuildContract, pkg: str) -> list:
    """Signals specific to python_cli archetype."""
    signals: list = []
    goals_lower = " ".join(contract.goals).lower()

    # CLI --help must work (either via python -m or via entry point)
    signals.append(CliExitSignal(
        description="CLI --help exits 0",
        command="python",
        args=["-m", pkg, "--help"],
        expect_exit=0,
    ))

    # If goals mention CLI, version, schema, validate, registry etc.
    # derive specific subcommand signals
    if "version" in goals_lower:
        signals.append(CliExitSignal(
            description="CLI version command works",
            command="python",
            args=["-m", pkg, "version"],
            expect_exit=0,
        ))

    if "schema" in goals_lower:
        # Use --help for the subcommand to avoid guessing positional vs flag
        signals.append(CliExitSignal(
            description="CLI schema subcommand exists",
            command="python",
            args=["-m", pkg, "schema", "--help"],
            expect_exit=0,
        ))

    if "validate" in goals_lower:
        signals.append(CliExitSignal(
            description="CLI validate subcommand exists",
            command="python",
            args=["-m", pkg, "validate", "--help"],
            expect_exit=0,
        ))

    if "registry" in goals_lower:
        signals.append(CliExitSignal(
            description="CLI registry subcommand exists",
            command="python",
            args=["-m", pkg, "registry", "--help"],
            expect_exit=0,
        ))

    return signals


def _fastapi_service_signals(contract: BuildContract, pkg: str) -> list:
    """Signals specific to fastapi_service archetype."""
    signals: list = []

    signals.append(FileExistsSignal(
        description="App module exists",
        file_path=f"{pkg}/app.py",
    ))

    signals.append(ImportCheckSignal(
        description="FastAPI is importable",
        module_name="fastapi",
    ))

    # App imports without error
    signals.append(CliExitSignal(
        description="App module imports successfully",
        command="python",
        args=["-c", f"from {pkg}.app import app"],
        expect_exit=0,
    ))

    return signals
