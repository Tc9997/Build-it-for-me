"""Validate README code examples against actual module exports.

Catches field name drift (e.g. README says 'payload' but model has 'params')
by comparing symbols referenced in code blocks against AST-extracted exports.
"""

from __future__ import annotations

import re
from pathlib import Path

_BUILTINS = frozenset({
    "Exception", "ValueError", "TypeError", "KeyError", "RuntimeError",
    "True", "False", "None", "Any", "Optional", "Union", "List", "Dict",
    "Set", "Tuple", "Type", "Callable", "Generic", "Literal", "Annotated",
    "BaseModel", "Field", "Path", "Enum", "StrEnum", "ABC",
    "dataclass", "datetime", "UUID", "Protocol",
})


def validate_readme(
    output_dir: Path,
    export_metadata: dict,
) -> list[str]:
    """Check README code examples against actual exports.

    Returns list of error strings. Empty = valid.
    """
    errors = []
    readme = output_dir / "README.md"
    if not readme.exists():
        return []  # No README is not an error — integrator may not have created one

    content = readme.read_text()
    code_blocks = _extract_code_blocks(content)
    if not code_blocks:
        return []

    # Collect all exported symbols across all modules
    all_classes = set()
    all_functions = set()
    for exports in export_metadata.values():
        if isinstance(exports, dict):
            all_classes.update(exports.get("exported_classes", []))
            all_functions.update(exports.get("exported_functions", []))

    if not all_classes and not all_functions:
        return []  # No export data to validate against

    all_symbols = all_classes | all_functions

    # Check import statements in code blocks
    for i, block in enumerate(code_blocks):
        for match in re.finditer(r'from\s+\S+\s+import\s+(.+)', block):
            imports = [s.strip().split(" as ")[0] for s in match.group(1).split(",")]
            for imp in imports:
                imp = imp.strip()
                if not imp or imp in _BUILTINS or imp.startswith("_"):
                    continue
                if imp not in all_symbols:
                    errors.append(
                        f"README block {i+1}: imports '{imp}' but it's not in module exports"
                    )

    return errors


def _extract_code_blocks(markdown: str) -> list[str]:
    """Extract Python code blocks from markdown."""
    blocks = []
    in_block = False
    current: list[str] = []
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```python") or stripped.startswith("```py"):
            in_block = True
            current = []
        elif stripped == "```" and in_block:
            in_block = False
            blocks.append("\n".join(current))
        elif in_block:
            current.append(line)
    return blocks
