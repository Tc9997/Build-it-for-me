"""Deterministic retry planner for freeform iteration.

Consumes the latest FreeformIssue from the journal and decides whether
a narrow, phase-local retry is worth attempting. All decisions are
code-driven — no LLM prompts.

Policy (v1):
  SETUP_FAILURE    -> retry_phase(setup), max 1 extra attempt
  TEST_FAILURE     -> retry_phase(test),  max 1 extra attempt
  Everything else  -> stop (not safe to retry automatically yet)

The retry planner does not execute retries. It returns a RetryDecision
that the orchestrator acts on.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from build_loop.freeform_journal import (
    FreeformJournal,
    FreeformIssue,
    IssueKind,
    IssueSource,
)


class RetryAction(str, Enum):
    """What the retry planner recommends."""
    RETRY_PHASE = "retry_phase"  # Re-run the failed phase
    STOP = "stop"                # Do not retry — stop the pipeline


class RetryDecision(BaseModel):
    """The output of the retry planner for one issue."""
    model_config = {"extra": "forbid"}

    action: RetryAction
    phase: str = Field(description="Phase to retry, or phase that caused the stop")
    rationale: str = Field(description="Why this decision was made")
    budget_remaining: int = Field(
        ge=0,
        description="How many more retries are available for this phase after this decision",
    )


# Phase -> max extra retry attempts allowed (beyond the initial attempt)
_RETRY_BUDGETS: dict[str, int] = {
    "setup": 1,
    "test": 1,
}

# Issue kinds that are safe to retry automatically
_RETRYABLE_KINDS: dict[IssueKind, str] = {
    IssueKind.SETUP_FAILURE: "setup",
    IssueKind.TEST_FAILURE: "test",
}


def plan_retry(issue: FreeformIssue, journal: FreeformJournal) -> RetryDecision:
    """Decide whether to retry based on the latest issue and journal history.

    Pure function: reads the issue and journal, returns a decision.
    Does not mutate anything.
    """
    target_phase = _RETRYABLE_KINDS.get(issue.kind)

    if target_phase is None:
        return RetryDecision(
            action=RetryAction.STOP,
            phase=issue.source.value,
            rationale=f"{issue.kind.value} is not automatically retryable",
            budget_remaining=0,
        )

    # Count how many times we've already attempted this phase
    phase_attempts = sum(1 for a in journal.attempts if a.phase == target_phase)
    max_attempts = 1 + _RETRY_BUDGETS.get(target_phase, 0)  # initial + retries
    remaining = max(0, max_attempts - phase_attempts)

    if remaining == 0:
        return RetryDecision(
            action=RetryAction.STOP,
            phase=target_phase,
            rationale=f"Retry budget exhausted for {target_phase} ({phase_attempts}/{max_attempts} attempts used)",
            budget_remaining=0,
        )

    return RetryDecision(
        action=RetryAction.RETRY_PHASE,
        phase=target_phase,
        rationale=f"{issue.kind.value} is retryable; {remaining} attempt(s) remaining",
        budget_remaining=remaining - 1,  # will be 0 after this retry
    )
