"""Acceptance agent: summarizes residual gaps after verification.

The verifier is the authority for machine-checkable pass/fail. This agent
does NOT overwrite verifier verdicts. It only:
  1. Reports the verifier's results
  2. Identifies residual gaps: acceptance criteria, behavioral expectations,
     and invariants that have no corresponding machine-checkable signal
  3. Uses LLM judgment ONLY for those uncovered residual items
"""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import AcceptanceResult, AcceptanceVerdict, BuildPlan, ExecResult


SYSTEM = """\
You are the Acceptance agent in an automated build system. An independent verifier \
has already executed machine-checkable signals and produced authoritative pass/fail results.

Your job is to assess ONLY the residual gaps — things the verifier could not check:
- Uncovered behavioral expectations
- Uncovered invariants
- Acceptance criteria that have no machine-checkable signal

You MUST respond with a single JSON object:

{
  "verdict": "pass" | "fail",
  "criteria_checked": ["list of ALL criteria assessed (verified + residual)"],
  "criteria_passed": ["things confirmed working (by verifier or by your assessment)"],
  "criteria_failed": ["things that don't work or are missing"],
  "notes": "string — overall assessment"
}

Rules:
- NEVER overwrite verifier results. If the verifier says a signal passed, it passed. \
  If the verifier says it failed, it failed.
- For verified signals: just carry them forward into criteria_checked and criteria_passed/failed.
- For uncovered behavioral expectations and invariants: assess them based on the project files \
  and note that they are "assessed by code review, not execution."
- verdict should be "fail" if ANY verifier signal failed OR if critical uncovered gaps exist.
- verdict should be "pass" only if all verifier signals passed AND no critical residual gaps.
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class AcceptanceAgent(Agent):
    name = "acceptance"
    system_prompt = SYSTEM
    model = "claude-haiku-4-5-20251001"  # Cheap — verifier is the authority, acceptance is residual

    def run(
        self,
        idea: str,
        plan: BuildPlan,
        project_files: dict[str, str],
        verification=None,
        smoke_result: ExecResult | None = None,
        require_verification: bool = True,
    ) -> AcceptanceResult:
        prompt_parts = [f"ORIGINAL IDEA:\n{idea}"]

        if verification:
            prompt_parts.append(
                f"\nVERIFICATION RESULT (authoritative — do not overwrite):\n"
                f"{json.dumps(verification.model_dump(), indent=2)}"
            )

        prompt_parts.append(
            f"\nPROJECT FILES:\n{json.dumps(project_files, indent=2)}"
        )

        if smoke_result:
            prompt_parts.append(
                f"\nSMOKE TEST:\n"
                f"Command: {smoke_result.command}\n"
                f"Exit code: {smoke_result.exit_code}\n"
                f"Stdout: {smoke_result.stdout[:1000]}\n"
                f"Stderr: {smoke_result.stderr[:1000]}"
            )

        data = self.call_json("\n".join(prompt_parts))
        result = AcceptanceResult(**data)

        # Hard constraint: if verifier failed, acceptance cannot pass
        if verification and not verification.passed:
            if result.verdict.value == "pass":
                result.verdict = AcceptanceVerdict.FAIL
                result.notes = (
                    f"OVERRIDDEN by verifier: {verification.summary}. "
                    f"LLM acceptance said pass but verifier is authoritative. "
                    f"Original notes: {result.notes}"
                )

        # Hard constraint: if verification was required but never run, verdict is incomplete
        if verification is None and require_verification:
            result.verdict = AcceptanceVerdict.INCOMPLETE
            result.notes = (
                f"Verification was skipped (degraded mode or not reached). "
                f"Cannot confirm pass without verifier results. "
                f"Original notes: {result.notes}"
            )

        self.log(
            f"verdict: {result.verdict} — "
            f"{len(result.criteria_passed)}/{len(result.criteria_checked)} criteria passed"
        )
        return result
