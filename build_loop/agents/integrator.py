"""Integrator agent: wires approved modules together and smoke-tests the result."""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import BuildArtifact, BuildPlan, IntegrationResult


SYSTEM = """\
You are the Integrator agent in an automated build system. You receive a set of approved \
module artifacts and wire them together into a working system.

You MUST respond with a single JSON object:

{
  "modules_integrated": ["module_id", ...],
  "success": true | false,
  "wiring_files": {"relative/path.py": "file contents", ...},
  "issues": ["string (any integration problems found)"],
  "notes": "string"
}

Your job:
1. Create any glue code needed to connect the modules (entry points, CLI, config loading, etc.)
2. Ensure imports between modules are correct and consistent.
3. Create a top-level entry point (main.py, __main__.py, or CLI) if one doesn't exist.
4. Add a README.md with setup and usage instructions.
5. Flag any integration issues (mismatched interfaces, circular deps, missing config).

If integration is clean, set success=true. If there are blocking issues, set success=false \
and describe them in issues.

Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class IntegratorAgent(Agent):
    name = "integrator"
    system_prompt = SYSTEM

    def run(self, plan: BuildPlan, artifacts: dict[str, BuildArtifact]) -> IntegrationResult:
        prompt = (
            f"PROJECT: {plan.project_name}\n"
            f"TECH STACK: {', '.join(plan.tech_stack)}\n"
            f"DIRECTORY STRUCTURE:\n{plan.directory_structure}\n\n"
            f"INTERFACES:\n{json.dumps([i.model_dump() for i in plan.interfaces], indent=2)}\n\n"
            f"APPROVED MODULES:\n{json.dumps({k: v.model_dump() for k, v in artifacts.items()}, indent=2)}"
        )

        data = self.call_json(prompt)
        result = IntegrationResult(**data)
        self.log(f"integrated {len(result.modules_integrated)} modules: "
                 + ("success" if result.success else f"FAILED ({len(result.issues)} issues)"))
        return result
