"""Structured issue and journal models for freeform iteration.

Freeform runs are best-effort. When they fail, we need machine-readable
diagnostics so future PRs can drive targeted patching, retries, or
escalation. This module defines the journal schema — the freeform
orchestrator writes to it, downstream tooling reads from it.

Design:
  - IssueKind: what category of failure
  - IssueSeverity: how bad
  - FreeformIssue: one discrete problem observed during the run
  - AttemptRecord: metadata for one attempt at a phase
  - FreeformJournal: ordered list of issues + attempts for the whole run
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class IssueKind(str, Enum):
    """Category of failure observed during a freeform run."""
    INTEGRATION_FAILURE = "integration_failure"
    SETUP_FAILURE = "setup_failure"
    TEST_FAILURE = "test_failure"
    ACCEPTANCE_FAILURE = "acceptance_failure"
    ACCEPTANCE_INCOMPLETE = "acceptance_incomplete"
    PIPELINE_ERROR = "pipeline_error"
    UNEXPECTED_CRASH = "unexpected_crash"


class IssueSeverity(str, Enum):
    """How severe the issue is."""
    BLOCKING = "blocking"   # Run cannot continue past this
    DEGRADED = "degraded"   # Run continued but quality is lower
    INFO = "info"           # Observed but not actionable yet


class IssueSource(str, Enum):
    """Which pipeline phase produced the issue."""
    RESEARCH = "research"
    PLAN = "plan"
    BUILD = "build"
    INTEGRATE = "integrate"
    WRITE = "write"
    SETUP = "setup"
    TEST = "test"
    OPTIMIZE = "optimize"
    ACCEPTANCE = "acceptance"


class FreeformIssue(BaseModel):
    """One discrete problem observed during a freeform run.

    Designed to be machine-readable so downstream tooling can
    filter, group, and act on issues programmatically.
    """
    model_config = {"extra": "forbid"}

    kind: IssueKind
    source: IssueSource
    severity: IssueSeverity
    summary: str = Field(description="Short human-readable description")
    detail: str = Field(default="", description="Longer context: stderr, traceback, etc.")
    retryable: bool = Field(
        default=False,
        description="Whether a targeted retry of this phase could fix the issue",
    )
    command: str = Field(default="", description="The command that failed, if applicable")
    verdict: str = Field(default="", description="Acceptance/test verdict, if applicable")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class AttemptRecord(BaseModel):
    """Metadata for one attempt at a pipeline phase.

    Captures what happened so later analysis can see the progression
    of attempts without re-running.
    """
    model_config = {"extra": "forbid"}

    phase: str = Field(description="Pipeline phase name (e.g. 'setup', 'test')")
    attempt_number: int = Field(ge=1, default=1)
    success: bool = False
    summary: str = Field(default="", description="What happened in this attempt")
    command: str = Field(default="", description="Command executed, if any")
    exit_code: int | None = Field(default=None, description="Exit code, if a command was run")
    issues_found: int = Field(default=0, description="Number of issues recorded in this attempt")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class FreeformJournal(BaseModel):
    """Ordered journal of issues and attempts for a freeform run.

    Persisted on BuildState so a failed run can be inspected later.
    The journal is append-only during a run — issues and attempts
    accumulate as the pipeline progresses.
    """
    model_config = {"extra": "forbid"}

    schema_version: Literal["1"] = "1"
    issues: list[FreeformIssue] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)

    @property
    def blocking_issues(self) -> list[FreeformIssue]:
        """Issues that prevented the run from completing."""
        return [i for i in self.issues if i.severity == IssueSeverity.BLOCKING]

    @property
    def retryable_issues(self) -> list[FreeformIssue]:
        """Issues that could potentially be fixed by a targeted retry."""
        return [i for i in self.issues if i.retryable]

    @property
    def has_blocking(self) -> bool:
        return any(i.severity == IssueSeverity.BLOCKING for i in self.issues)

    def record_issue(
        self,
        kind: IssueKind,
        source: IssueSource,
        severity: IssueSeverity,
        summary: str,
        *,
        detail: str = "",
        retryable: bool = False,
        command: str = "",
        verdict: str = "",
    ) -> FreeformIssue:
        """Create and append an issue. Returns it for convenience."""
        issue = FreeformIssue(
            kind=kind,
            source=source,
            severity=severity,
            summary=summary,
            detail=detail,
            retryable=retryable,
            command=command,
            verdict=verdict,
        )
        self.issues.append(issue)
        return issue

    def record_attempt(
        self,
        phase: str,
        success: bool,
        *,
        attempt_number: int = 1,
        summary: str = "",
        command: str = "",
        exit_code: int | None = None,
    ) -> AttemptRecord:
        """Create and append an attempt record. Returns it for convenience."""
        # Count issues recorded since the last attempt for this phase
        issues_found = sum(
            1 for i in self.issues
            if i.source.value == phase
        )
        # Subtract issues from prior attempts at same phase
        prior_issues = sum(
            a.issues_found for a in self.attempts if a.phase == phase
        )
        new_issues = max(0, issues_found - prior_issues)

        attempt = AttemptRecord(
            phase=phase,
            attempt_number=attempt_number,
            success=success,
            summary=summary,
            command=command,
            exit_code=exit_code,
            issues_found=new_issues,
        )
        self.attempts.append(attempt)
        return attempt

    def summary_text(self) -> str:
        """One-paragraph summary for CLI output."""
        total = len(self.issues)
        blocking = len(self.blocking_issues)
        retryable = len(self.retryable_issues)
        attempts = len(self.attempts)
        if total == 0:
            return f"Clean run: {attempts} phase attempts, no issues recorded."
        parts = [f"{total} issue(s)"]
        if blocking:
            parts.append(f"{blocking} blocking")
        if retryable:
            parts.append(f"{retryable} retryable")
        parts.append(f"{attempts} phase attempt(s)")
        return "Journal: " + ", ".join(parts) + "."
