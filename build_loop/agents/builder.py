"""Builder agent: takes a module spec + interfaces and produces code.

Receives deterministic context about already-built dependencies:
actual file paths, exported symbols, and compact source snippets.
Also receives static framework-version guidance to prevent
outdated API patterns.
"""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.analysis.exports import ModuleExports
from build_loop.analysis.framework_hints import get_framework_hints
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
- When DEPENDENCY CONTEXT is provided, use the ACTUAL export names and file paths \
  from already-built dependencies — do NOT guess or invent symbol names.
- Write real, runnable code. No stubs, no TODOs, no placeholders.
- Write tests that verify the module satisfies its interface contract.
- Keep dependencies minimal. Only import what's in the tech stack.
- Follow all FRAMEWORK RULES exactly — violations will fail review.
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
        dependency_exports: dict[str, ModuleExports] | None = None,
    ) -> BuildArtifact:
        # Gather relevant interfaces
        relevant = [i for i in plan.interfaces
                    if i.name in module.interfaces_provided or i.name in module.interfaces_consumed]

        prompt_parts = [
            f"PROJECT: {plan.project_name}",
            f"TECH STACK: {', '.join(plan.tech_stack)}",
            f"DIRECTORY STRUCTURE:\n{plan.directory_structure}",
        ]

        # Framework version guidance (deterministic, static)
        hints = get_framework_hints(plan.tech_stack)
        if hints:
            prompt_parts.append(f"\n{hints}")

        prompt_parts.append(
            f"\nINTERFACE CONTRACTS:\n{json.dumps([i.model_dump() for i in relevant], indent=2)}"
        )

        # Dependency context: actual exports from already-built dependencies
        if dependency_exports:
            dep_context = _format_dependency_context(dependency_exports)
            if dep_context:
                prompt_parts.append(f"\nDEPENDENCY CONTEXT (use these ACTUAL names):\n{dep_context}")

        prompt_parts.append(
            f"\nMODULE TO BUILD:\n{json.dumps(module.model_dump(), indent=2)}"
        )

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


def _format_dependency_context(dep_exports: dict[str, ModuleExports]) -> str:
    """Format dependency exports as compact structured context for the builder."""
    parts = []
    for dep_id, exports in dep_exports.items():
        lines = [f"  Module: {dep_id}"]
        lines.append(f"    Files: {exports.files}")
        if exports.exported_classes:
            lines.append(f"    Classes: {exports.exported_classes}")
        if exports.exported_functions:
            lines.append(f"    Functions: {exports.exported_functions}")
        if exports.exported_constants:
            lines.append(f"    Constants: {exports.exported_constants}")
        if exports.import_statements:
            lines.append(f"    Imports: {exports.import_statements[:10]}")
        parts.append("\n".join(lines))

    return "\n".join(parts)
