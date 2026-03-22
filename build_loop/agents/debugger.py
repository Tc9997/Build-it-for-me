"""Debugger agent: takes execution errors and produces fixes.

Inputs are truncated deterministically to fit context windows:
- stdout/stderr: last 3000 chars (tail, where errors usually are)
- project files: only files < 10k chars, total budget 200k chars
- previous fixes: last 2 only
"""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import BuildPlan, DebugFix, ExecResult

_MAX_OUTPUT_CHARS = 3000
_MAX_FILE_CHARS = 10_000
_MAX_TOTAL_FILE_CHARS = 200_000
_MAX_PREVIOUS_FIXES = 2


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
- Do NOT modify template-locked files (pyproject.toml, .gitignore, conftest.py). \
  Only modify builder-owned and generated files.
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
        # Truncate error output to tail (where errors usually are)
        stdout = error.stdout[-_MAX_OUTPUT_CHARS:] if error.stdout else ""
        stderr = error.stderr[-_MAX_OUTPUT_CHARS:] if error.stderr else ""

        # Budget project files: skip large files, cap total
        budgeted_files = _budget_project_files(project_files)

        prompt_parts = [
            f"PROJECT: {plan.project_name}",
            f"TECH STACK: {', '.join(plan.tech_stack)}",
            f"\nFAILED COMMAND: {error.command}",
            f"EXIT CODE: {error.exit_code}",
        ]
        if stdout:
            prompt_parts.append(f"STDOUT (last {_MAX_OUTPUT_CHARS} chars):\n{stdout}")
        if stderr:
            prompt_parts.append(f"STDERR (last {_MAX_OUTPUT_CHARS} chars):\n{stderr}")

        prompt_parts.append(f"\nPROJECT FILES:\n{json.dumps(budgeted_files, indent=2)}")

        # Only include last N previous fixes to bound context
        if previous_fixes:
            recent = previous_fixes[-_MAX_PREVIOUS_FIXES:]
            prompt_parts.append(
                f"\nPREVIOUS FIX ATTEMPTS (last {len(recent)}):\n"
                + json.dumps([f.model_dump() for f in recent], indent=2)
            )

        data = self.call_json("\n".join(prompt_parts))
        fix = DebugFix(**data)
        self.log(f"diagnosis: {fix.diagnosis[:100]}...")
        self.log(f"  fixing {len(fix.file_changes)} files"
                 + (f", adding deps: {fix.new_dependencies}" if fix.new_dependencies else ""))
        return fix


def _budget_project_files(files: dict[str, str]) -> dict[str, str]:
    """Select project files that fit within the context budget.

    Prioritizes: Python files over others, smaller files first.
    Skips files > _MAX_FILE_CHARS. Caps total at _MAX_TOTAL_FILE_CHARS.
    """
    # Sort: .py first, then by size ascending
    sorted_files = sorted(
        files.items(),
        key=lambda kv: (0 if kv[0].endswith(".py") else 1, len(kv[1])),
    )

    result = {}
    total = 0
    for path, content in sorted_files:
        if len(content) > _MAX_FILE_CHARS:
            continue
        if total + len(content) > _MAX_TOTAL_FILE_CHARS:
            break
        result[path] = content
        total += len(content)

    return result
