"""Reviewer agent: reviews builder output against spec and interfaces.

After deterministic screening passes, reviewer gets only the code files
and concise metadata — not the full artifact JSON blob.

Uses a cheaper model (haiku) since deterministic checks already passed.
"""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import BuildArtifact, BuildPlan, ModuleSpec, ReviewResult, ReviewVerdict


SYSTEM = """\
You are a Reviewer agent in an automated build system. You review code produced by a Builder agent \
and decide whether it meets the specification. Syntax and import checks have already passed.

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
2. Are there logic bugs beyond what syntax checking catches?
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
    model = "claude-haiku-4-5-20251001"  # Cheaper model — deterministic checks already passed

    def run(self, module: ModuleSpec, artifact: BuildArtifact, plan: BuildPlan) -> ReviewResult:
        relevant = [i for i in plan.interfaces
                    if i.name in module.interfaces_provided or i.name in module.interfaces_consumed]

        # Compact prompt: code files + concise metadata, not full JSON blobs
        prompt_parts = [
            f"MODULE: {module.id} — {module.description[:300]}",
        ]
        if relevant:
            prompt_parts.append(
                f"\nINTERFACES:\n{json.dumps([{'name': i.name, 'description': i.description} for i in relevant], indent=2)}"
            )

        for path, content in artifact.files.items():
            prompt_parts.append(f"\n--- {path} ---\n{content}")
        for path, content in artifact.tests.items():
            prompt_parts.append(f"\n--- {path} (test) ---\n{content}")

        data = self.call_json("\n".join(prompt_parts))
        result = ReviewResult(**data)
        self.log(f"reviewed {module.id}: {result.verdict.value}"
                 + (f" ({len(result.issues)} issues)" if result.issues else ""))
        return result
