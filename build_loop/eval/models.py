"""Eval harness data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalTask(BaseModel):
    """A single eval task from the corpus."""
    id: str
    name: str
    archetype: str
    idea: str
    expected_signals: list[dict] = Field(default_factory=list)
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    """Result of running a single eval task.

    passed is determined by eval_verify (corpus signals), NOT by the
    build system's internal verifier or acceptance. Both modes are
    scored identically.
    """
    task_id: str
    task_name: str
    archetype: str
    mode: str  # "template_first" or "freeform"

    # Eval-scored (the authority)
    passed: bool = False
    signal_results: list[dict] = Field(
        default_factory=list,
        description="Results from eval_verify running corpus expected_signals"
    )

    # Pipeline metadata
    pipeline_completed: bool = False
    debug_rounds: int = 0
    wall_time_seconds: float = 0.0
    error: str = ""

    # Self-reported (for analysis, not scoring)
    verification_passed: bool | None = None
    acceptance_verdict: str = ""


class EvalSuiteResult(BaseModel):
    """Aggregated results for an eval suite run."""
    mode: str
    total_tasks: int = 0
    tasks_passed: int = 0
    tasks_failed: int = 0
    tasks_errored: int = 0
    pass_rate: float = 0.0
    total_wall_time: float = 0.0
    avg_debug_rounds: float = 0.0
    results: list[EvalRunResult] = Field(default_factory=list)
