"""Derive base success signals deterministically from contract archetype + goals.

These are merged with (not replacing) LLM-generated signals from the spec
compiler. Derived signals cover archetype-level guarantees. LLM signals
cover project-specific deliverables the archetype can't know about.

CLI checks use the console_scripts entry point from the template, not
python -m, because not all packages have __main__.py.
"""

from __future__ import annotations

from build_loop.contract import (
    BuildContract,
    CliExitSignal,
    FileExistsSignal,
    ImportCheckSignal,
)


def derive_signals(contract: BuildContract) -> list:
    """Derive base signals from contract archetype and goals.

    Returns deterministic signals for archetype-level guarantees only.
    These should be MERGED with project-specific LLM signals, not replace them.
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
    """Signals specific to python_cli archetype.

    Does NOT assume python -m works (requires __main__.py).
    Only checks file existence and importability — runtime CLI checks
    are handled by the archetype verifier pack which reads pyproject.toml.
    """
    signals: list = []

    # Package __init__.py exists
    signals.append(FileExistsSignal(
        description="Package __init__.py exists",
        file_path=f"{pkg}/__init__.py",
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

    return signals


def merge_signals(derived: list, llm_generated: list) -> list:
    """Merge derived base signals with LLM-generated project-specific signals.

    Derived signals come first (archetype guarantees).
    LLM signals are appended if they don't duplicate a derived signal.
    Deduplication is by (type, description) pair.
    """
    seen = set()
    merged = []

    for s in derived:
        key = (s.type, s.description)
        if key not in seen:
            seen.add(key)
            merged.append(s)

    for s in llm_generated:
        key = (s.type, s.description)
        if key not in seen:
            # Validate LLM signal structure before including
            if hasattr(s, "command") and hasattr(s, "args"):
                # Skip malformed CLI signals (spaces in command)
                if " " in getattr(s, "command", ""):
                    continue
            seen.add(key)
            merged.append(s)

    return merged
