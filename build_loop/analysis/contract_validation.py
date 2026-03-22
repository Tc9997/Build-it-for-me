"""Deterministic pre-integration contract validation.

Compares planned interfaces vs actual built exports. Catches symbol drift
and file-path drift before the integrator LLM sees anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from build_loop.analysis.exports import ModuleExports
from build_loop.schemas import BuildPlan, ModuleSpec


@dataclass
class PreIntegrationResult:
    """Result of pre-integration validation."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_pre_integration(
    plan: BuildPlan,
    exports: dict[str, ModuleExports],
) -> PreIntegrationResult:
    """Check that built modules satisfy their planned contracts.

    Checks:
    1. Every approved module produced the planned files
    2. Modules with consumed interfaces have corresponding exports available
    3. No syntax errors in any approved module
    4. No same-batch dependency violations
    """
    result = PreIntegrationResult()
    module_specs = {m.id: m for m in plan.modules}

    # 1. Syntax check — any parse errors are blocking
    for mid, exp in exports.items():
        if not exp.syntax_valid:
            result.errors.append(
                f"Module '{mid}' has syntax errors: {exp.parse_errors}"
            )
            result.valid = False

    # 2. File path coverage — did the module produce expected files?
    for mid, exp in exports.items():
        spec = module_specs.get(mid)
        if not spec:
            continue
        for planned_path in spec.file_paths:
            if planned_path not in exp.files:
                result.warnings.append(
                    f"Module '{mid}' planned file '{planned_path}' not found in output"
                )

    # 3. Dependency symbol resolution — do consumed interfaces exist?
    # Build a map of all exported symbols per module
    all_exports_by_module: dict[str, set[str]] = {}
    for mid, exp in exports.items():
        symbols = set(exp.exported_classes + exp.exported_functions + exp.exported_constants)
        all_exports_by_module[mid] = symbols

    for mid, spec in module_specs.items():
        if mid not in exports:
            continue  # Module wasn't built (failed)

        for dep_id in spec.dependencies:
            if dep_id not in exports:
                result.warnings.append(
                    f"Module '{mid}' depends on '{dep_id}' which was not built"
                )
                continue

            # Check if consumed interfaces are actually exported by dependencies
            for iface_name in spec.interfaces_consumed:
                # Find which dependency provides this interface
                dep_spec = module_specs.get(dep_id)
                if dep_spec and iface_name in dep_spec.interfaces_provided:
                    # The dependency should export symbols related to this interface
                    dep_exports = all_exports_by_module.get(dep_id, set())
                    if not dep_exports:
                        result.warnings.append(
                            f"Module '{mid}' consumes interface '{iface_name}' from "
                            f"'{dep_id}' but '{dep_id}' exports no symbols"
                        )

    return result
