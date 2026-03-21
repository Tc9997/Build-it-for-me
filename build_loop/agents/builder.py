"""Builder agent: takes a module spec + interfaces and produces code."""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import BuildArtifact, BuildPlan, ModuleSpec, ReviewResult


SYSTEM = """\
You are a Builder agent in an automated build system. You receive a module specification \
and interface contracts, and you produce working code.

You MUST respond with a single JSON object:

{
  "module_id": "string",
  "files": {"relative/path.py": "file contents as string", ...},
  "tests": {"tests/test_module.py": "test file contents as string", ...},
  "notes": "string (any notes about implementation decisions)"
}

Rules:
- Implement EXACTLY the interfaces specified. Do not invent new public interfaces.
- Import shared types from wherever the interface contract says they live.
- Write real, runnable code. No stubs, no TODOs, no placeholders.
- Write tests that verify the module satisfies its interface contract.
- Keep dependencies minimal. Only import what's in the tech stack.
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class BuilderAgent(Agent):
    name = "builder"
    system_prompt = SYSTEM

    def run(
        self,
        module: ModuleSpec,
        plan: BuildPlan,
        revision_feedback: ReviewResult | None = None,
    ) -> BuildArtifact:
        # Gather relevant interfaces
        relevant = [i for i in plan.interfaces
                    if i.name in module.interfaces_provided or i.name in module.interfaces_consumed]

        prompt_parts = [
            f"PROJECT: {plan.project_name}",
            f"TECH STACK: {', '.join(plan.tech_stack)}",
            f"DIRECTORY STRUCTURE:\n{plan.directory_structure}",
            f"\nINTERFACE CONTRACTS:\n{json.dumps([i.model_dump() for i in relevant], indent=2)}",
            f"\nMODULE TO BUILD:\n{json.dumps(module.model_dump(), indent=2)}",
        ]

        if revision_feedback:
            prompt_parts.append(
                f"\nREVISION REQUESTED — fix these issues:\n"
                f"Issues: {revision_feedback.issues}\n"
                f"Suggestions: {revision_feedback.suggestions}\n"
                f"Notes: {revision_feedback.notes}"
            )

        data = self.call_json("\n".join(prompt_parts))
        artifact = BuildArtifact(**data)
        self.log(f"built {module.id}: {len(artifact.files)} files, {len(artifact.tests)} test files")
        return artifact
