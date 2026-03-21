"""Planner agent: decomposes a contract + research into a BuildPlan.

The planner receives a structured BuildContract (not raw prose) and must
produce a plan that demonstrably covers every contract goal. The output
includes goals_covered and non_goals_acknowledged so coverage can be
validated without LLM judgment.
"""

from __future__ import annotations

from build_loop.agents.base import Agent
from build_loop.schemas import BuildPlan


SYSTEM = """\
You are the Planner agent in an automated build system. You receive a structured \
BuildContract and research findings, and produce a detailed build plan.

You MUST respond with a single JSON object matching this schema:

{
  "schema_version": "1",
  "project_name": "string",
  "description": "string",

  "contract_hash": "string — copy the contract_hash value provided in the prompt",
  "goals_covered": {
    "exact goal string from contract": ["module_id_1", "module_id_2"],
    "another goal string": ["module_id_3"]
  },
  "non_goals_acknowledged": ["exact non-goal string from contract", ...],

  "tech_stack": ["string — exact pip/npm package names"],
  "directory_structure": "string (tree-style text)",
  "interfaces": [
    {
      "name": "string",
      "description": "string",
      "inputs": {"param": "type/desc"},
      "outputs": {"param": "type/desc"},
      "file_path": "string (relative from project root)",
      "code": "string (real, runnable interface code)"
    }
  ],
  "modules": [
    {
      "id": "string (snake_case)",
      "name": "string",
      "description": "string — be VERY specific",
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

CRITICAL RULES:
- goals_covered MUST map EVERY goal from the contract to at least one module ID. \
  Use the EXACT goal strings from the contract as keys. If you cannot cover a goal, \
  explain why in the module description but still map it.
- non_goals_acknowledged MUST list EVERY non-goal from the contract. Copy them exactly.
- contract_hash: copy the value provided in the prompt exactly.
- USE the research findings for library choices and approaches.
- Define interfaces FIRST, then modules that implement/consume them.
- build_order: list of batches. Modules in the same batch run in parallel.
- First batch should always be shared schemas/types.
- ALWAYS include a requirements.txt in one of the module's file_paths.
- Module descriptions should be detailed enough for a builder to implement unambiguously.
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


FREEFORM_SYSTEM = """\
You are the Planner agent in an experimental freeform build system. You receive a project \
idea and research findings, and produce a build plan.

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
      "code": "string (real, runnable interface code)"
    }
  ],
  "modules": [
    {
      "id": "string (snake_case)",
      "name": "string",
      "description": "string — be VERY specific",
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
- USE the research findings for library choices and approaches.
- Define interfaces FIRST, then modules that implement/consume them.
- build_order: list of batches. Modules in the same batch run in parallel.
- First batch should always be shared schemas/types.
- ALWAYS include a requirements.txt in one of the module's file_paths.
- Module descriptions should be detailed enough for a builder to implement unambiguously.
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class PlannerAgent(Agent):
    name = "planner"
    system_prompt = SYSTEM

    def run(self, plan_context: str) -> BuildPlan:
        data = self.call_json(plan_context)
        plan = BuildPlan(**data)
        self.log(f"planned {len(plan.modules)} modules in {len(plan.build_order)} batches")
        if plan.goals_covered:
            self.log(f"  goals covered: {len(plan.goals_covered)}")
        return plan


class FreeformPlannerAgent(Agent):
    """Planner for freeform mode — no contract, no coverage requirements."""
    name = "planner"
    system_prompt = FREEFORM_SYSTEM

    def run(self, plan_context: str) -> BuildPlan:
        data = self.call_json(plan_context)
        plan = BuildPlan(**data)
        self.log(f"planned {len(plan.modules)} modules in {len(plan.build_order)} batches")
        return plan
