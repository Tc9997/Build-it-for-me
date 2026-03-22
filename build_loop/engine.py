"""Engine abstraction: interface that all build engines must satisfy.

Both TemplateFirstOrchestrator and FreeformOrchestrator already satisfy
this interface implicitly. This module makes the contract explicit so
the router can depend on the abstraction, not on concrete classes.

This is a seam, not a rewrite. Existing orchestrators are not modified.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from build_loop.schemas import BuildState


class PromiseLevel(str, Enum):
    """What quality guarantees an engine provides."""
    VERIFIED = "verified"        # Verifier-backed, deterministic gates
    BEST_EFFORT = "best_effort"  # LLM-judged, experimental


class EngineCapabilities(BaseModel):
    """What an engine can do, declared upfront."""
    model_config = {"extra": "forbid"}

    name: str
    promise_level: PromiseLevel
    supports_resume: bool = False
    supports_verify: bool = False
    supported_archetypes: list[str] = Field(default_factory=list)
    description: str = ""


class BuildEngine(ABC):
    """Abstract interface for a build engine.

    Both template_first and freeform already have run() and state.
    This ABC makes that contract explicit.
    """

    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Declare what this engine can do."""
        ...

    @abstractmethod
    def run(self, idea: str) -> str:
        """Run the full build pipeline. Returns the output directory."""
        ...

    @property
    @abstractmethod
    def state(self) -> BuildState:
        """Current build state."""
        ...


class RouteOrigin(str, Enum):
    """How the engine was selected."""
    EXPLICIT = "explicit"    # User chose this mode via --mode flag
    DEFAULT = "default"      # No --mode flag — used the default mode
    DEGRADED = "degraded"    # Originally another mode, fell back due to policy/error


class RouteDecision(BaseModel):
    """The result of selecting an engine for a build request.

    Captures which engine was selected, why, with what confidence,
    and how the selection was made. Auditable and testable.
    """
    model_config = {"extra": "forbid"}

    engine_name: str
    promise_level: PromiseLevel
    origin: RouteOrigin = RouteOrigin.DEFAULT
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    rationale: str = ""
    mode_value: str = ""

    @property
    def summary(self) -> str:
        """One-line human-readable summary for CLI output."""
        promise = "verified" if self.promise_level == PromiseLevel.VERIFIED else "best-effort"
        return f"Engine: {self.engine_name} | Promise: {promise} | Origin: {self.origin.value}"
