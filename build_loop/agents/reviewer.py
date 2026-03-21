"""Reviewer agent: reviews builder output against spec and interfaces."""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import BuildArtifact, BuildPlan, ModuleSpec, ReviewResult, ReviewVerdict


SYSTEM = """\
You are a Reviewer agent in an automated build system. You review code produced by a Builder agent \
and decide whether it meets the specification.

You MUST respond with a single JSON object:

{
  "module_id": "string",
  "verdict": "approve" | "revise",
  "issues": ["string (specific problems that MUST be fixed)"],
  "suggestions": ["string (optional improvements)"],
  "notes": "string"
}

Review criteria — only flag real problems:
1. Does the code implement the required interfaces correctly?
2. Does the code actually work (no syntax errors, broken imports, logic bugs)?
3. Are there tests, and do they test the interface contract?
4. Any security issues (hardcoded secrets, injection, unsafe deserialization)?
5. Does it stay within the specified tech stack?

Do NOT nitpick style, naming conventions, or minor preferences. Only REVISE if there are \
genuine correctness, completeness, or security issues. If the code works and satisfies the \
interface, APPROVE it.

Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class ReviewerAgent(Agent):
    name = "reviewer"
    system_prompt = SYSTEM

    def run(self, module: ModuleSpec, artifact: BuildArtifact, plan: BuildPlan) -> ReviewResult:
        relevant = [i for i in plan.interfaces
                    if i.name in module.interfaces_provided or i.name in module.interfaces_consumed]

        prompt = (
            f"MODULE SPEC:\n{json.dumps(module.model_dump(), indent=2)}\n\n"
            f"INTERFACE CONTRACTS:\n{json.dumps([i.model_dump() for i in relevant], indent=2)}\n\n"
            f"BUILDER OUTPUT:\n{json.dumps(artifact.model_dump(), indent=2)}"
        )

        data = self.call_json(prompt)
        result = ReviewResult(**data)
        self.log(f"reviewed {module.id}: {result.verdict.value}"
                 + (f" ({len(result.issues)} issues)" if result.issues else ""))
        return result
