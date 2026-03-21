"""Acceptance tester: validates the built project against the original intent."""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import AcceptanceResult, BuildPlan, ExecResult


SYSTEM = """\
You are the Acceptance Tester agent in an automated build system. Your job is to determine \
whether a completed project actually fulfills the user's original intent.

You receive:
- The original idea/request
- The build plan
- The project files
- Test results and execution output

You MUST respond with a single JSON object:

{
  "verdict": "pass" | "fail",
  "criteria_checked": ["list of things you verified"],
  "criteria_passed": ["things that work correctly"],
  "criteria_failed": ["things that don't work or are missing"],
  "notes": "string — overall assessment and any caveats"
}

Rules:
- Break the original idea into concrete, testable acceptance criteria.
- Be practical — if the user asked for "a bot to scrape wine auctions", check that:
  - There's actual scraping logic targeting a real site
  - Data is parsed into a usable format
  - There's some recommendation/scoring mechanism
  - The code runs without errors
- Don't fail for cosmetic issues. Fail for: missing core functionality, broken execution, \
  wrong output format, missing critical integration.
- If tests pass but the code clearly doesn't do what was asked, that's a FAIL.
- If the code works but lacks polish (no CLI help text, basic error messages), that's still a PASS.
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class AcceptanceAgent(Agent):
    name = "acceptance"
    system_prompt = SYSTEM

    def run(
        self,
        idea: str,
        plan: BuildPlan,
        project_files: dict[str, str],
        test_result: ExecResult | None = None,
        smoke_result: ExecResult | None = None,
    ) -> AcceptanceResult:
        prompt_parts = [
            f"ORIGINAL IDEA:\n{idea}",
            f"\nBUILD PLAN:\n{json.dumps(plan.model_dump(), indent=2)}",
            f"\nPROJECT FILES:\n{json.dumps(project_files, indent=2)}",
        ]

        if test_result:
            prompt_parts.append(
                f"\nTEST RESULTS:\n"
                f"Command: {test_result.command}\n"
                f"Exit code: {test_result.exit_code}\n"
                f"Stdout: {test_result.stdout}\n"
                f"Stderr: {test_result.stderr}"
            )

        if smoke_result:
            prompt_parts.append(
                f"\nSMOKE TEST:\n"
                f"Command: {smoke_result.command}\n"
                f"Exit code: {smoke_result.exit_code}\n"
                f"Stdout: {smoke_result.stdout}\n"
                f"Stderr: {smoke_result.stderr}"
            )

        data = self.call_json("\n".join(prompt_parts))
        result = AcceptanceResult(**data)
        self.log(
            f"verdict: {result.verdict.value} — "
            f"{len(result.criteria_passed)}/{len(result.criteria_checked)} criteria passed"
        )
        return result
