"""SpecCompiler agent: turns (idea + research) into a structured BuildContract.

LLM-powered. Converts prose into the contract that everything downstream
consumes. The contract must be specific enough for the planner, verifier,
and acceptance tester to work from.

EnvironmentSnapshot and Policy are deterministic (no LLM). Research and
SpecCompiler are LLM-backed. This is the boundary.
"""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.contract import BuildContract, SCHEMA_VERSION


SYSTEM = """\
You are the SpecCompiler in an automated build system. You receive a project idea \
and research findings, and produce a structured BuildContract.

You MUST respond with a single JSON object matching this schema:

{
  "schema_version": "1",
  "project_name": "string (snake_case)",
  "summary": "string — one paragraph describing what the project does",
  "archetype": "python_cli | fastapi_service | unsupported — python_cli for CLI tools, scripts, data pipelines. fastapi_service for REST APIs, web services, bots with HTTP interfaces. Use 'unsupported' if the project does NOT fit either archetype (e.g. mobile apps, browser extensions, hardware projects, game engines).",
  "goals": ["concrete things the project MUST do — be specific and testable"],
  "non_goals": ["things explicitly OUT of scope — the planner should NOT build these"],
  "constraints": ["hard requirements: language, framework, licensing, performance bounds"],
  "target_runtime": "string — e.g. python3.11+",
  "run_mode": "batch | service",
  "capability_requirements": [
    {
      "type": "docker | network | hardware | service | system_tool",
      "name": "human-readable name, e.g. Redis, ffmpeg",
      "required": true,
      "affects_phases": ["setup", "test", "optimize"]
    }
  ],
  "secrets_required": ["ENV_VAR_NAME — API keys, tokens that must be set"],
  "acceptance_criteria": ["human-readable criteria for 'done'"],
  "success_signals": [
    {
      "type": "cli_exit | http_probe | file_exists | stdout_contains | import_check | schema_valid",
      "description": "what this checks",
      "command": "MUST be a single executable with NO spaces (e.g. 'python'). NEVER 'python -m foo'.",
      "args": ["each arg is a separate token. Examples: args=['-m', 'pkg.cli', 'version'] or args=['-m', 'pkg.cli', 'schema', '--model', 'Foo']"],
      "expect_exit": 0,
      "expect_contains": "optional string",
      "path": "optional — for http_probe",
      "method": "GET",
      "expect_status": 200,
      "expect_body_contains": "optional",
      "file_path": "optional — for file_exists",
      "module_name": "optional — for import_check"
    }
  ],
  "behavioral_expectations": [
    {
      "description": "what behavior this tests",
      "given": "input or precondition",
      "expect": "expected output or side effect",
      "verifiable": true
    }
  ],
  "invariants": [
    {
      "description": "something that must NEVER happen",
      "category": "correctness | security | data-integrity | performance"
    }
  ],
  "open_questions": ["things you genuinely cannot determine — be honest"]
}

Rules:
- Goals must be concrete and testable. "Build a scraper" is too vague. \
  "Scrape auction listings from acker.com including lot number, title, estimate, and current bid" is good.
- Non-goals prevent scope creep. If someone asks for a wine scraper, "building a recommendation engine" \
  might be a non-goal unless they explicitly asked for it.
- Success signals must be machine-checkable. Include at least:
  - One cli_exit or import_check (does the code run at all?)
  - One stdout_contains or http_probe (does it produce expected output?)
- Behavioral expectations should cover the core use case with concrete examples.
- Invariants should cover security (no hardcoded secrets, no SQL injection) and data integrity.
- run_mode: use "service" for servers, bots, watchers, anything long-running. "batch" for everything else.
- Be honest about open_questions. If the idea mentions an API but doesn't specify which one, ask.
- Use the research findings — they tell you what libraries, APIs, and approaches are realistic.
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class SpecCompilerAgent(Agent):
    name = "spec_compiler"
    system_prompt = SYSTEM

    def run(self, idea: str, research_json: str = "") -> BuildContract:
        prompt = f"PROJECT IDEA:\n{idea}"
        if research_json:
            prompt += f"\n\nRESEARCH FINDINGS:\n{research_json}"

        data = self.call_json(prompt)
        # Ensure schema version is set
        data["schema_version"] = SCHEMA_VERSION
        contract = BuildContract(**data)
        self.log(
            f"compiled contract: {len(contract.goals)} goals, "
            f"{len(contract.success_signals)} signals, "
            f"{len(contract.invariants)} invariants"
        )
        return contract
