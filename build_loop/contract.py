"""BuildContract: the structured specification that replaces raw prose.

The SpecCompiler (LLM-powered) turns a user's idea + research into a
BuildContract. Everything downstream — planner, verifier, acceptance —
consumes the contract, not the original prose.

Schema version is explicit so journals, caches, and templates can detect
incompatible contracts without silent misinterpretation.

All models use extra="forbid" — unknown fields are rejected, not silently
dropped. SuccessSignal is a discriminated union: each signal type has its
own model with only the fields that type requires.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Strict base — all contract models reject unknown fields
# ---------------------------------------------------------------------------

class StrictModel(BaseModel):
    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Success signals — discriminated union of per-type models
# ---------------------------------------------------------------------------

class CliExitSignal(StrictModel):
    """Run a command and check exit code."""
    type: Literal["cli_exit"] = "cli_exit"
    description: str
    command: str
    args: list[str] = Field(default_factory=list)
    expect_exit: int = 0


class StdoutContainsSignal(StrictModel):
    """Run a command and check stdout contains a string."""
    type: Literal["stdout_contains"] = "stdout_contains"
    description: str
    command: str
    args: list[str] = Field(default_factory=list)
    expect_contains: str


class HttpProbeSignal(StrictModel):
    """Hit an HTTP endpoint and check status/body."""
    type: Literal["http_probe"] = "http_probe"
    description: str
    path: str
    method: str = "GET"
    expect_status: int = 200
    expect_body_contains: str = ""


class FileExistsSignal(StrictModel):
    """A file must exist after run."""
    type: Literal["file_exists"] = "file_exists"
    description: str
    file_path: str


class ImportCheckSignal(StrictModel):
    """A Python module must be importable."""
    type: Literal["import_check"] = "import_check"
    description: str
    module_name: str


class SchemaValidSignal(StrictModel):
    """Output must match a JSON schema."""
    type: Literal["schema_valid"] = "schema_valid"
    description: str
    command: str = Field(description="Command whose stdout is validated")
    args: list[str] = Field(default_factory=list)
    json_schema: dict


SuccessSignal = Annotated[
    Union[
        CliExitSignal,
        StdoutContainsSignal,
        HttpProbeSignal,
        FileExistsSignal,
        ImportCheckSignal,
        SchemaValidSignal,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Behavioral expectations — richer than liveness
# ---------------------------------------------------------------------------

class BehavioralExpectation(StrictModel):
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

class Invariant(StrictModel):
    """A property that must hold at all times."""
    description: str
    category: str = Field(
        default="correctness",
        description="e.g. correctness, security, data-integrity, performance"
    )


# ---------------------------------------------------------------------------
# Capability requirements — typed external dependencies
# ---------------------------------------------------------------------------

class CapabilityType(str, Enum):
    """Types of external capabilities a project may require."""
    DOCKER = "docker"         # Needs Docker to run containers
    NETWORK = "network"       # Needs outbound network access
    HARDWARE = "hardware"     # Needs specific hardware (GPIO, Bluetooth, GPU)
    SERVICE = "service"       # Needs an external service running (DB, cache, queue)
    SYSTEM_TOOL = "system_tool"  # Needs a CLI tool installed (ffmpeg, wkhtmltopdf)


class CapabilityRequirement(StrictModel):
    """A typed external dependency.

    Policy matches on `type`, not on substring search through free text.
    `affects_phases` declares which pipeline phases are blocked without
    this capability.
    """
    type: CapabilityType
    name: str = Field(description="Human-readable name, e.g. 'Redis', 'ffmpeg'")
    required: bool = Field(default=True, description="True = hard requirement, False = optional enhancement")
    affects_phases: list[str] = Field(
        default_factory=lambda: ["setup", "test", "optimize"],
        description="Pipeline phases blocked if this capability is missing"
    )


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------

class BuildContract(StrictModel):
    """The structured specification for a build.

    Produced by the SpecCompiler from (idea + research). Consumed by planner,
    verifier, and acceptance. This is the single source of truth for what
    the project must do.

    extra="forbid" — unknown fields are rejected.
    schema_version is Literal["1"] — mismatches fail validation.
    """
    schema_version: Literal["1"] = SCHEMA_VERSION

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

    # Capability requirements (typed, not prose)
    capability_requirements: list[CapabilityRequirement] = Field(
        default_factory=list,
        description="Structured external dependencies with typed capabilities"
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
