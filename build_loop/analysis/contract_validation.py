"""Deterministic pre-integration contract validation.

Compares planned interfaces vs actual built exports. Catches symbol drift,
file-path drift, and same-batch dependency violations before the integrator
LLM sees anything.

Uses production exports only — test-file symbols are excluded.
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
    1. Syntax errors in any approved module → error (blocking)
    2. Missing planned production files → error (blocking)
    3. Same-batch dependency violations → error (blocking)
    4. Unbuilt dependencies → warning
    5. Dependency with no production exports → warning
    """
    result = PreIntegrationResult()
    module_specs = {m.id: m for m in plan.modules}

    # Build batch position map for same-batch detection
    batch_of: dict[str, int] = {}
    for batch_idx, batch_ids in enumerate(plan.build_order):
        for mid in batch_ids:
            batch_of[mid] = batch_idx

    # 1. Syntax check — blocking
    for mid, exp in exports.items():
        if not exp.syntax_valid:
            result.errors.append(
                f"Module '{mid}' has syntax errors: {exp.parse_errors}"
            )
            result.valid = False

    # 2. Missing planned production files — blocking
    for mid, exp in exports.items():
        spec = module_specs.get(mid)
        if not spec:
            continue
        for planned_path in spec.file_paths:
            if planned_path not in exp.files:
                result.errors.append(
                    f"Module '{mid}' planned production file '{planned_path}' "
                    f"not found in output. Actual files: {exp.files}"
                )
                result.valid = False

    # 3. Same-batch dependency violations (direct + transitive) — blocking
    for mid, spec in module_specs.items():
        if mid not in exports:
            continue
        my_batch = batch_of.get(mid)
        if my_batch is None:
            continue
        # Check direct dependencies
        for dep_id in spec.dependencies:
            dep_batch = batch_of.get(dep_id)
            if dep_batch is not None and dep_batch == my_batch:
                result.errors.append(
                    f"Module '{mid}' depends on '{dep_id}' but both are in "
                    f"batch {my_batch + 1}. Dependency context is unavailable "
                    f"for same-batch modules."
                )
                result.valid = False
        # Check transitive: if A depends on B and B depends on C,
        # and A and C are in the same batch, A won't have C's context
        for dep_id in spec.dependencies:
            dep_spec = module_specs.get(dep_id)
            if not dep_spec:
                continue
            for transitive_id in dep_spec.dependencies:
                trans_batch = batch_of.get(transitive_id)
                if trans_batch is not None and trans_batch == my_batch and transitive_id != mid:
                    result.warnings.append(
                        f"Module '{mid}' transitively depends on '{transitive_id}' "
                        f"(via '{dep_id}') but both are in batch {my_batch + 1}."
                    )

    # 4. Unbuilt dependency — warning
    for mid, spec in module_specs.items():
        if mid not in exports:
            continue
        for dep_id in spec.dependencies:
            if dep_id not in exports:
                result.warnings.append(
                    f"Module '{mid}' depends on '{dep_id}' which was not built"
                )

    # 5. Dependency with no production exports — warning
    all_prod_exports: dict[str, set[str]] = {}
    for mid, exp in exports.items():
        symbols = set(exp.exported_classes + exp.exported_functions + exp.exported_constants)
        all_prod_exports[mid] = symbols

    for mid, spec in module_specs.items():
        if mid not in exports:
            continue
        for dep_id in spec.dependencies:
            if dep_id in all_prod_exports and not all_prod_exports[dep_id]:
                result.warnings.append(
                    f"Module '{mid}' depends on '{dep_id}' but '{dep_id}' "
                    f"has no production exports"
                )

    return result
