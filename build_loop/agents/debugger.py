"""Debugger agent: takes execution errors and produces fixes."""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import BuildPlan, DebugFix, ExecResult


SYSTEM = """\
You are the Debugger agent in an automated build system. You receive error output from \
running code and must produce a fix.

You MUST respond with a single JSON object:

{
  "diagnosis": "string — what went wrong and why",
  "file_changes": {"relative/path.py": "FULL new file contents (not a diff)", ...},
  "new_dependencies": ["any new pip/npm packages needed"],
  "notes": "string"
}

Rules:
- Read the error carefully. Most errors are: import errors, missing deps, wrong paths, \
  type errors, or missing config.
- When you change a file, include the COMPLETE new file contents, not a patch.
- Only change files that are directly related to the error. Don't refactor unrelated code.
- If the error is a missing dependency, add it to new_dependencies.
- If the error is a missing environment variable or config file, create the config with \
  sensible defaults.
- If the error suggests a fundamental design flaw, say so in diagnosis — but still provide \
  the best fix you can.
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class DebuggerAgent(Agent):
    name = "debugger"
    system_prompt = SYSTEM

    def run(
        self,
        error: ExecResult,
        plan: BuildPlan,
        project_files: dict[str, str],
        previous_fixes: list[DebugFix] | None = None,
    ) -> DebugFix:
        prompt_parts = [
            f"PROJECT: {plan.project_name}",
            f"TECH STACK: {', '.join(plan.tech_stack)}",
            f"\nFAILED COMMAND: {error.command}",
            f"EXIT CODE: {error.exit_code}",
            f"STDOUT:\n{error.stdout}" if error.stdout else "",
            f"STDERR:\n{error.stderr}" if error.stderr else "",
            f"\nPROJECT FILES:\n{json.dumps(project_files, indent=2)}",
        ]

        if previous_fixes:
            prompt_parts.append(
                f"\nPREVIOUS FIX ATTEMPTS (these didn't fully work):\n"
                + json.dumps([f.model_dump() for f in previous_fixes], indent=2)
            )

        data = self.call_json("\n".join(prompt_parts))
        fix = DebugFix(**data)
        self.log(f"diagnosis: {fix.diagnosis[:100]}...")
        self.log(f"  fixing {len(fix.file_changes)} files"
                 + (f", adding deps: {fix.new_dependencies}" if fix.new_dependencies else ""))
        return fix
