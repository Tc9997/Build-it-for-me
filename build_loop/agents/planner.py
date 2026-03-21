"""Planner agent: decomposes an idea + research into a BuildPlan with interfaces and modules."""

from __future__ import annotations

from build_loop.agents.base import Agent
from build_loop.schemas import BuildPlan


SYSTEM = """\
You are the Planner agent in an automated build system. You receive a project idea along with \
research findings (libraries, APIs, feasibility) and produce a detailed, structured build plan.

You MUST respond with a single JSON object matching this schema:

{
  "project_name": "string",
  "description": "string",
  "tech_stack": ["string — exact pip/npm package names"],
  "directory_structure": "string (tree-style text)",
  "interfaces": [
    {
      "name": "string",
      "description": "string",
      "inputs": {"param": "type/desc"},
      "outputs": {"param": "type/desc"},
      "file_path": "string (relative from project root)",
      "code": "string (real, runnable interface code — Pydantic models, dataclasses, protocols, etc.)"
    }
  ],
  "modules": [
    {
      "id": "string (snake_case)",
      "name": "string",
      "description": "string — be VERY specific about what this module does and how",
      "size": "S|M|L",
      "dependencies": ["module_id"],
      "interfaces_provided": ["interface_name"],
      "interfaces_consumed": ["interface_name"],
      "file_paths": ["string"]
    }
  ],
  "build_order": [["batch_1_ids"], ["batch_2_ids"], ...],
  "setup_commands": [
    "python3 -m venv .venv",
    ".venv/bin/pip install --upgrade pip",
    ".venv/bin/pip install -r requirements.txt"
  ],
  "test_command": "pytest -v",
  "run_command": "python main.py"
}

Rules:
- USE the research findings. The researcher already identified the best libraries and approaches. \
  Don't ignore that work.
- Define interfaces FIRST, then modules that implement/consume them.
- build_order: list of batches. Modules in the same batch run in parallel. Respect dependencies.
- First batch should always be shared schemas/types.
- Keep modules small and focused.
- file_paths must be concrete relative paths from project root.
- interface "code" must be real, runnable Python (Pydantic models, dataclasses, Protocol classes).
- setup_commands: include venv creation, pip install, and any external service setup.
- ALWAYS include a requirements.txt in one of the module's file_paths listing ALL pip dependencies.
- test_command: how to run tests (use pytest unless there's a reason not to).
- run_command: how to run the finished project.
- Module descriptions should be detailed enough that a builder can implement without ambiguity. \
  Include specific library usage, API endpoints, data formats.
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class PlannerAgent(Agent):
    name = "planner"
    system_prompt = SYSTEM

    def run(self, idea_with_research: str) -> BuildPlan:
        data = self.call_json(idea_with_research)
        plan = BuildPlan(**data)
        self.log(f"planned {len(plan.modules)} modules in {len(plan.build_order)} batches")
        return plan
