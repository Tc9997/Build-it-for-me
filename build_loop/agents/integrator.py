"""Integrator agent: wires approved modules together.

Receives the ownership manifest and export metadata so it knows which
files already exist (from template) and what symbols each module exports.
"""

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
1. Create any glue code needed to connect the modules (entry points, config loading, etc.)
2. Ensure imports between modules are correct and consistent.
3. Create a top-level entry point if one doesn't exist.
4. Add a README.md with setup and usage instructions.
5. Update pyproject.toml with correct dependencies and entry points.
6. Flag any integration issues (mismatched interfaces, circular deps, missing config).

CRITICAL RULES:
- Do NOT create files that already exist in TEMPLATE FILES. Those are managed by the template.
- Do NOT create files that builder modules already produced. Check EXISTING FILES.
- pyproject.toml [project.scripts] must point at the ACTUAL CLI module, not a placeholder.
  Look at EXPORT METADATA to find which module exports a main() function.
- README examples must use the ACTUAL field names from the built models, not guesses.

If integration is clean, set success=true. If there are blocking issues, set success=false \
and describe them in issues.

Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class IntegratorAgent(Agent):
    name = "integrator"
    system_prompt = SYSTEM

    def run(
        self,
        plan: BuildPlan,
        artifacts: dict[str, BuildArtifact],
        template_files: list[str] | None = None,
        export_metadata: dict | None = None,
    ) -> IntegrationResult:
        # Collect all files already produced by builders
        existing_files = set()
        for artifact in artifacts.values():
            existing_files.update(artifact.files.keys())
            existing_files.update(artifact.tests.keys())

        prompt_parts = [
            f"PROJECT: {plan.project_name}",
            f"TECH STACK: {', '.join(plan.tech_stack)}",
            f"DIRECTORY STRUCTURE:\n{plan.directory_structure}",
        ]

        if template_files:
            prompt_parts.append(
                f"\nTEMPLATE FILES (already exist, do NOT recreate):\n"
                f"{json.dumps(sorted(template_files), indent=2)}"
            )

        prompt_parts.append(
            f"\nEXISTING FILES (produced by builders, do NOT recreate):\n"
            f"{json.dumps(sorted(existing_files), indent=2)}"
        )

        if export_metadata:
            # Compact export summary
            export_summary = {}
            for mid, exports in export_metadata.items():
                if isinstance(exports, dict):
                    export_summary[mid] = {
                        "files": exports.get("files", []),
                        "classes": exports.get("exported_classes", []),
                        "functions": exports.get("exported_functions", []),
                    }
            prompt_parts.append(
                f"\nEXPORT METADATA (actual symbols per module):\n"
                f"{json.dumps(export_summary, indent=2)}"
            )

        prompt_parts.append(
            f"\nINTERFACES:\n{json.dumps([i.model_dump() for i in plan.interfaces], indent=2)}"
        )

        # Send only file paths, not full content (too large)
        module_summary = {}
        for mid, artifact in artifacts.items():
            module_summary[mid] = {
                "files": list(artifact.files.keys()),
                "tests": list(artifact.tests.keys()),
            }
        prompt_parts.append(
            f"\nAPPROVED MODULES:\n{json.dumps(module_summary, indent=2)}"
        )

        data = self.call_json("\n".join(prompt_parts))
        result = IntegrationResult(**data)
        self.log(f"integrated {len(result.modules_integrated)} modules: "
                 + ("success" if result.success else f"FAILED ({len(result.issues)} issues)"))
        return result
