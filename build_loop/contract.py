"""BuildContract: the structured specification that replaces raw prose.

The SpecCompiler (LLM-powered) turns a user's idea + research into a
BuildContract. Everything downstream — planner, verifier, acceptance —
consumes the contract, not the original prose.

Schema version is explicit so journals, caches, and templates can detect
incompatible contracts without silent misinterpretation.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Success signals — machine-checkable assertions
# ---------------------------------------------------------------------------

class SuccessSignalType(str, Enum):
    """Types of verifiable success signals."""
    CLI_EXIT = "cli_exit"           # run a command, check exit code
    HTTP_PROBE = "http_probe"       # hit an endpoint, check status/body
    FILE_EXISTS = "file_exists"     # a file must exist after run
    STDOUT_CONTAINS = "stdout_contains"  # command output contains a string
    IMPORT_CHECK = "import_check"   # a Python module must be importable
    SCHEMA_VALID = "schema_valid"   # output matches a JSON schema


class SuccessSignal(BaseModel):
    """A single machine-checkable assertion derived from the contract."""
    type: SuccessSignalType
    description: str = Field(description="Human-readable description of what this checks")
    # CLI_EXIT / STDOUT_CONTAINS
    command: str = ""
    args: list[str] = Field(default_factory=list)
    expect_exit: int = 0
    expect_contains: str = ""
    # HTTP_PROBE
    path: str = ""
    method: str = "GET"
    expect_status: int = 200
    expect_body_contains: str = ""
    # FILE_EXISTS
    file_path: str = ""
    # IMPORT_CHECK
    module_name: str = ""
    # SCHEMA_VALID
    json_schema: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Behavioral expectations — richer than liveness
# ---------------------------------------------------------------------------

class BehavioralExpectation(BaseModel):
    """A concrete expected behavior: given input X, expect output matching Y."""
    description: str
    given: str = Field(description="Input or precondition")
    expect: str = Field(description="Expected output, side effect, or state change")
    verifiable: bool = Field(
        default=True,
        description="Can this be checked programmatically?"
    )


# ---------------------------------------------------------------------------
# Invariant — things that must NEVER happen
# ---------------------------------------------------------------------------

class Invariant(BaseModel):
    """A property that must hold at all times."""
    description: str
    category: str = Field(
        default="correctness",
        description="e.g. correctness, security, data-integrity, performance"
    )


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------

class BuildContract(BaseModel):
    """The structured specification for a build.

    Produced by the SpecCompiler from (idea + research). Consumed by planner,
    verifier, and acceptance. This is the single source of truth for what
    the project must do.
    """
    schema_version: str = SCHEMA_VERSION

    # Identity
    project_name: str
    summary: str = Field(description="One-paragraph description of what the project does")

    # Scope
    goals: list[str] = Field(description="Concrete things the project MUST do")
    non_goals: list[str] = Field(
        default_factory=list,
        description="Explicitly out of scope — planner won't build, verifier won't test"
    )

    # Constraints
    constraints: list[str] = Field(
        default_factory=list,
        description="Hard requirements: language, framework, licensing, etc."
    )
    target_runtime: str = Field(
        default="python3.11+",
        description="Minimum runtime environment"
    )
    run_mode: Literal["batch", "service"] = Field(
        default="batch",
        description="'batch' for CLI/scripts, 'service' for servers/bots/watchers"
    )

    # Dependencies
    external_dependencies: list[str] = Field(
        default_factory=list,
        description="APIs, databases, Docker images, hardware — things outside pip"
    )
    secrets_required: list[str] = Field(
        default_factory=list,
        description="Environment variables / API keys that must be present"
    )

    # Verification
    acceptance_criteria: list[str] = Field(
        description="Human-readable criteria for 'done'"
    )
    success_signals: list[SuccessSignal] = Field(
        default_factory=list,
        description="Machine-checkable assertions for the verifier"
    )
    behavioral_expectations: list[BehavioralExpectation] = Field(
        default_factory=list,
        description="Given X, expect Y — richer than liveness checks"
    )
    invariants: list[Invariant] = Field(
        default_factory=list,
        description="Things that must NEVER happen"
    )

    # Open questions — things the system couldn't resolve
    open_questions: list[str] = Field(default_factory=list)
