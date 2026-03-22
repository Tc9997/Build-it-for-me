"""Shared pipeline operations used by both template_first and freeform modes.

Contains: build+review loop, write project, setup environment, test+debug,
optimize, and shared helpers. Mode-specific orchestration logic lives in
the mode modules, not here.
"""

from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from build_loop.agents.builder import BuilderAgent
from build_loop.agents.reviewer import ReviewerAgent
from build_loop.agents.integrator import IntegratorAgent
from build_loop.agents.executor import ExecutorAgent
from build_loop.agents.debugger import DebuggerAgent
from build_loop.agents.optimizer import OptimizerAgent
from build_loop.analysis.exports import ModuleExports, analyze_artifact
from build_loop.analysis.contract_validation import validate_pre_integration
from build_loop.safety import safe_output_path
from build_loop.schemas import (
    AcceptanceVerdict,
    BuildArtifact,
    BuildPlan,
    BuildState,
    DebugFix,
    ModuleSpec,
    ReviewResult,
    ReviewVerdict,
    TaskStatus,
)

console = Console()

# Lock for thread-safe state mutations during parallel builds
_state_lock = threading.Lock()

MAX_REVIEW_REVISIONS = 3
MAX_DEBUG_ROUNDS = 5


class ModuleRejectedError(Exception):
    """Raised when a module exhausts review revisions without approval."""

    def __init__(self, module_id: str, final_review: ReviewResult):
        self.module_id = module_id
        self.final_review = final_review
        super().__init__(
            f"Module '{module_id}' rejected after {MAX_REVIEW_REVISIONS} revisions: "
            f"{final_review.issues}"
        )


class IntegrationFailedError(Exception):
    """Raised when integration reports blocking issues."""


class PipelineError(Exception):
    """Raised for any phase-gate failure that should stop the pipeline."""


def log(name: str, msg: str) -> None:
    console.print(f"[bold cyan][{name}][/bold cyan] {msg}")


def phase(num: str, name: str, desc: str) -> None:
    console.print(Panel(desc, title=f"[bold blue]PHASE {num}: {name}[/bold blue]"))


def build_all(
    state: BuildState,
    builder: BuilderAgent,
    reviewer: ReviewerAgent,
) -> None:
    """Build all modules batch by batch with parallel execution within batches.

    After each approved module:
      - Runs deterministic export analysis (AST-based)
      - Stores export metadata in state
    Downstream batches receive actual dependency exports as builder context.

    After all batches: runs pre-integration contract validation.
    """
    plan = state.plan
    if not plan:
        raise PipelineError("No plan")

    module_specs = {m.id: m for m in plan.modules}
    # Accumulates exports as modules are approved
    all_exports: dict[str, ModuleExports] = {}

    for batch_idx, batch in enumerate(plan.build_order):
        console.print(f"\n[bold green]  Batch {batch_idx + 1}/{len(plan.build_order)}:[/bold green] {batch}")

        modules = {m.id: m for m in plan.modules if m.id in batch}

        with ThreadPoolExecutor(max_workers=max(len(modules), 1)) as pool:
            futures = {
                pool.submit(
                    build_and_review, mod, plan, state, builder, reviewer,
                    _gather_dependency_exports(mod, module_specs, all_exports),
                ): mod.id
                for mod in modules.values()
            }
            for future in as_completed(futures):
                mid = futures[future]
                try:
                    artifact, final_review = future.result()
                    # Thread-safe state mutations
                    with _state_lock:
                        state.artifacts[mid] = artifact
                        modules[mid].status = TaskStatus.APPROVED

                        # Deterministic export analysis on approved artifact
                        exports = analyze_artifact(artifact)
                        all_exports[mid] = exports
                        state.module_exports[mid] = exports.model_dump()
                    log("analysis", f"{mid}: {len(exports.exported_classes)} classes, "
                        f"{len(exports.exported_functions)} functions"
                        + (" [syntax errors]" if not exports.syntax_valid else ""))

                except ModuleRejectedError as e:
                    console.print(f"  [bold red]{mid} REJECTED: {e.final_review.issues}[/bold red]")
                    with _state_lock:
                        modules[mid].status = TaskStatus.FAILED
                except Exception as e:
                    console.print(f"  [bold red]{mid} failed: {e}[/bold red]")
                    with _state_lock:
                        modules[mid].status = TaskStatus.FAILED

    approved = [m for m in plan.modules if m.status == TaskStatus.APPROVED]
    if not approved:
        raise PipelineError("No modules passed review — cannot proceed to integration")

    failed = [m for m in plan.modules if m.status == TaskStatus.FAILED]
    if failed:
        console.print(
            f"\n  [yellow]Warning: {len(failed)} module(s) rejected: "
            f"{[m.id for m in failed]}[/yellow]"
        )

    # Pre-integration contract validation (deterministic)
    console.print("\n  [bold]Pre-integration validation...[/bold]")
    pre_result = validate_pre_integration(plan, all_exports)
    for w in pre_result.warnings:
        console.print(f"  [dim]Warning: {w}[/dim]")
    if not pre_result.valid:
        for e in pre_result.errors:
            console.print(f"  [bold red]Error: {e}[/bold red]")
        raise PipelineError(
            f"Pre-integration validation failed: {pre_result.errors}"
        )
    console.print("  [green]Pre-integration checks passed[/green]")


def _gather_dependency_exports(
    module: ModuleSpec,
    module_specs: dict[str, ModuleSpec],
    all_exports: dict[str, ModuleExports],
) -> dict[str, ModuleExports]:
    """Gather export metadata for a module's direct dependencies.

    Only includes dependencies that have already been built and analyzed
    (from earlier batches). Same-batch dependencies won't be here.
    """
    dep_exports = {}
    for dep_id in module.dependencies:
        if dep_id in all_exports:
            dep_exports[dep_id] = all_exports[dep_id]
    return dep_exports


def build_and_review(
    module: ModuleSpec,
    plan: BuildPlan,
    state: BuildState,
    builder: BuilderAgent,
    reviewer: ReviewerAgent,
    dependency_exports: dict[str, ModuleExports] | None = None,
) -> tuple[BuildArtifact, ReviewResult]:
    """Build → syntax screen → review → revise loop.

    Before LLM review, runs deterministic syntax check on builder output.
    If syntax is broken, feeds that as revision feedback without wasting
    a reviewer round.
    """
    module.status = TaskStatus.IN_PROGRESS
    artifact = builder.run(
        module, plan,
        dependency_exports=dependency_exports or {},
    )
    last_review = None

    for attempt in range(MAX_REVIEW_REVISIONS):
        # Deterministic syntax screening before LLM review
        # Pass known project module paths so cross-module imports aren't flagged
        syntax_issues = _screen_syntax(artifact, _collect_known_modules(plan, state))
        if syntax_issues:
            log("pipeline", f"{module.id}: syntax issues found, requesting revision")
            module.status = TaskStatus.REVISION
            # Feed syntax errors as revision feedback without LLM review
            syntax_feedback = ReviewResult(
                module_id=module.id,
                verdict=ReviewVerdict.REVISE,
                issues=syntax_issues,
                notes="Deterministic syntax check failed — fix before review",
            )
            with _state_lock:
                state.reviews.setdefault(module.id, []).append(syntax_feedback)
            artifact = builder.run(
                module, plan,
                revision_feedback=syntax_feedback,
                dependency_exports=dependency_exports or {},
            )
            continue

        module.status = TaskStatus.IN_REVIEW
        review = reviewer.run(module, artifact, plan)
        with _state_lock:
            state.reviews.setdefault(module.id, []).append(review)
        last_review = review

        if review.verdict == ReviewVerdict.APPROVE:
            return artifact, review

        log("pipeline", f"{module.id}: revision {attempt + 1}/{MAX_REVIEW_REVISIONS}")
        module.status = TaskStatus.REVISION
        artifact = builder.run(
            module, plan,
            revision_feedback=review,
            dependency_exports=dependency_exports or {},
        )

    # If all rounds were syntax failures, last_review is None.
    # Create a synthetic rejection review from the last syntax feedback.
    if last_review is None:
        reviews = state.reviews.get(module.id, [])
        if reviews:
            last_review = reviews[-1]
        else:
            last_review = ReviewResult(
                module_id=module.id,
                verdict=ReviewVerdict.REVISE,
                issues=["Module exhausted all revision rounds without passing syntax checks"],
            )

    raise ModuleRejectedError(module.id, last_review)


def _screen_syntax(
    artifact: BuildArtifact,
    known_project_modules: frozenset[str] | None = None,
) -> list[str]:
    """Cheap deterministic screening on builder output.

    Checks:
    1. AST parse errors (syntax)
    2. Unresolved internal imports (project modules that don't exist
       in this artifact OR in the broader project)

    known_project_modules: module paths from other built artifacts and
    the plan's file structure. Prevents false positives on cross-module
    imports within the same project.
    """
    exports = analyze_artifact(artifact)
    issues = []
    if not exports.syntax_valid:
        issues.extend(f"Syntax error: {e}" for e in exports.parse_errors)

    # Filter unresolved imports against known project modules
    if exports.unresolved_imports and known_project_modules:
        truly_unresolved = []
        for imp in exports.unresolved_imports:
            # Extract module path from import statement
            if imp.startswith("from "):
                parts = imp.split()
                module_path = parts[1] if len(parts) > 1 else ""
            elif imp.startswith("import "):
                parts = imp.split()
                module_path = parts[1].split(",")[0].strip() if len(parts) > 1 else ""
            else:
                truly_unresolved.append(imp)
                continue

            # Check if any prefix of the module path is in known modules
            parts = module_path.split(".")
            found = False
            for i in range(len(parts), 0, -1):
                if ".".join(parts[:i]) in known_project_modules:
                    found = True
                    break
            if not found:
                truly_unresolved.append(imp)

        if truly_unresolved:
            issues.extend(f"Unresolved import: {imp}" for imp in truly_unresolved)
    elif exports.unresolved_imports:
        # No known modules — report all unresolved
        issues.extend(f"Unresolved import: {imp}" for imp in exports.unresolved_imports)

    return issues


def _collect_known_modules(plan: BuildPlan, state: BuildState) -> frozenset[str]:
    """Collect all known module paths from the plan and already-built artifacts.

    This allows the syntax screener to recognize cross-module imports within
    the same project as valid.
    """
    modules = set()

    # From plan's file structure — all planned file paths
    if plan:
        for mod_spec in plan.modules:
            for path in mod_spec.file_paths:
                if path.endswith(".py"):
                    parts = path.replace("/", ".").removesuffix(".py").split(".")
                    for i in range(len(parts)):
                        modules.add(".".join(parts[:i + 1]))
                    if parts[-1] == "__init__":
                        modules.add(".".join(parts[:-1]))

    # From already-built artifacts
    for artifact in state.artifacts.values():
        for path in list(artifact.files.keys()) + list(artifact.tests.keys()):
            if path.endswith(".py"):
                parts = path.replace("/", ".").removesuffix(".py").split(".")
                for i in range(len(parts)):
                    modules.add(".".join(parts[:i + 1]))
                if parts[-1] == "__init__":
                    modules.add(".".join(parts[:-1]))
                # src/ layout: also add paths without the "src" prefix
                # so "from mypackage.foo import X" works when file is src/mypackage/foo.py
                if parts[0] == "src" and len(parts) > 1:
                    sub = parts[1:]
                    for i in range(len(sub)):
                        modules.add(".".join(sub[:i + 1]))
                    if sub[-1] == "__init__":
                        modules.add(".".join(sub[:-1]))

    # From plan's directory structure (often contains package names)
    if plan and plan.directory_structure:
        for line in plan.directory_structure.split("\n"):
            line = line.strip().strip("├─│└ ")
            if line.endswith("/"):
                pkg = line.rstrip("/").replace("/", ".")
                if pkg:
                    modules.add(pkg)

    return frozenset(modules)


def write_project(
    state: BuildState,
    output_dir: str,
    safe_write_fn,
) -> None:
    """Write all generated files to the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files_written = 0

    if state.plan:
        for iface in state.plan.interfaces:
            if iface.code:
                safe_write_fn(iface.file_path, iface.code)
                files_written += 1

    # Collect all artifact files. Detect duplicate paths across builders.
    # Integration wiring is the sole owner of shared files (pyproject.toml, README, etc.)
    # and is allowed to overwrite artifact files.
    all_files: dict[str, str] = {}
    path_owners: dict[str, str] = {}  # path → module_id for conflict detection

    for mid in sorted(state.artifacts.keys()):
        artifact = state.artifacts[mid]
        for path, content in artifact.files.items():
            if path in path_owners:
                raise PipelineError(
                    f"Duplicate file path '{path}' produced by both "
                    f"'{path_owners[path]}' and '{mid}'. "
                    f"Builders must not write to shared files."
                )
            all_files[path] = content
            path_owners[path] = mid
        for path, content in artifact.tests.items():
            if path in path_owners:
                raise PipelineError(
                    f"Duplicate test path '{path}' produced by both "
                    f"'{path_owners[path]}' and '{mid}'."
                )
            all_files[path] = content
            path_owners[path] = mid

    # Integration wiring may overwrite builder files only for known shared
    # metadata files. Other overwrites are flagged.
    _INTEGRATOR_OWNED = {"pyproject.toml", "setup.py", "setup.cfg", "README.md",
                         "requirements.txt", "Makefile", "Dockerfile", ".gitignore",
                         "__init__.py", "__main__.py"}
    if state.integration and state.integration.wiring_files:
        for path, content in state.integration.wiring_files.items():
            basename = path.rsplit("/", 1)[-1] if "/" in path else path
            if path in path_owners and basename not in _INTEGRATOR_OWNED:
                log("pipeline", f"  [yellow]Warning: integrator overwrites builder file '{path}' (from {path_owners[path]})[/yellow]")
            all_files[path] = content

    for path, content in all_files.items():
        safe_write_fn(path, content)
        files_written += 1

    log("pipeline", f"wrote {files_written} files to {out}")


def setup_environment(state: BuildState, executor: ExecutorAgent, venv_cmd_fn) -> None:
    """Run project setup commands. Raises PipelineError if critical setup fails."""
    plan = state.plan
    if not plan or not plan.setup_commands:
        default_setup = [
            "python3 -m venv .venv",
            ".venv/bin/pip install --upgrade pip",
        ]
        req_path = Path(executor.project_dir) / "requirements.txt"
        pyproject_path = Path(executor.project_dir) / "pyproject.toml"
        if req_path.exists():
            default_setup.append(".venv/bin/pip install -r requirements.txt")
        if pyproject_path.exists():
            default_setup.append(".venv/bin/pip install -e .")
        default_setup.append(".venv/bin/pip install pytest")
        results = executor.setup_project(default_setup)
        state.exec_history.extend(results)
        failed = [r for r in results if not r.success]
        if failed:
            raise PipelineError(
                f"Setup failed: {failed[0].command} — {failed[0].stderr[-200:]}"
            )
        return

    results = executor.setup_project(plan.setup_commands)
    state.exec_history.extend(results)
    failed = [r for r in results if not r.success]
    if failed:
        raise PipelineError(
            f"Setup failed: {failed[0].command} — {failed[0].stderr[-200:]}"
        )


def test_and_debug_loop(
    state: BuildState,
    executor: ExecutorAgent,
    debugger: DebuggerAgent,
    venv_cmd_fn,
    safe_write_fn,
    read_files_fn,
) -> None:
    """Run tests → debug → fix → rerun, up to MAX_DEBUG_ROUNDS."""
    plan = state.plan
    test_cmd = venv_cmd_fn(plan.test_command) if plan else "pytest -v"
    previous_fixes: list[DebugFix] = []

    for round_num in range(MAX_DEBUG_ROUNDS):
        state.debug_rounds = round_num + 1
        console.print(f"\n  [bold]Debug round {round_num + 1}/{MAX_DEBUG_ROUNDS}[/bold]")

        test_result = executor.run_tests(test_cmd)
        state.exec_history.append(test_result)

        if test_result.success:
            console.print("  [bold green]All tests pass![/bold green]")
            return

        project_files = read_files_fn()
        try:
            fix = debugger.run(
                error=test_result,
                plan=plan,
                project_files=project_files,
                previous_fixes=previous_fixes if previous_fixes else None,
            )
        except Exception as e:
            console.print(f"  [yellow]Debugger failed: {e} — skipping round[/yellow]")
            continue
        previous_fixes.append(fix)
        apply_fix(fix, executor, venv_cmd_fn, safe_write_fn, state)

    raise PipelineError(f"Tests did not pass after {MAX_DEBUG_ROUNDS} debug rounds")


def optimize(
    state: BuildState,
    executor: ExecutorAgent,
    optimizer: OptimizerAgent,
    debugger: DebuggerAgent,
    venv_cmd_fn,
    safe_write_fn,
    read_files_fn,
) -> None:
    """Run the optimizer on working code, then re-verify tests still pass."""
    plan = state.plan
    project_files = read_files_fn()

    test_results = [r for r in state.exec_history if r.success and ("test" in r.command.lower() or "pytest" in r.command.lower())]
    test_result = test_results[-1] if test_results else None

    # Budget project files for optimizer context
    from build_loop.agents.debugger import _budget_project_files
    budgeted_files = _budget_project_files(project_files)

    try:
        result = optimizer.run(plan=plan, project_files=budgeted_files, test_result=test_result)
    except Exception as e:
        console.print(f"  [yellow]Optimizer failed: {e} — skipping optimization[/yellow]")
        return

    file_changes = result.get("file_changes", {})
    if not file_changes:
        console.print("  [dim]No optimizations needed[/dim]")
        return

    state.optimization_count = len(result.get("optimizations", []))

    for path, content in file_changes.items():
        safe_write_fn(path, content)
        log("optimizer", f"  optimized {path}")

    install_dependencies(
        result.get("new_dependencies", []), executor, venv_cmd_fn, state
    )

    console.print("\n  [bold]Re-running tests after optimization...[/bold]")
    test_cmd = venv_cmd_fn(plan.test_command) if plan else "pytest -v"
    verify = executor.run_tests(test_cmd)
    state.exec_history.append(verify)

    if verify.success:
        console.print("  [bold green]Tests still pass after optimization[/bold green]")
    else:
        console.print("  [yellow]Optimization broke tests — entering debug loop to fix...[/yellow]")
        previous_fixes: list[DebugFix] = []
        for attempt in range(3):
            current_files = read_files_fn()
            try:
                fix = debugger.run(
                    error=verify, plan=plan, project_files=current_files,
                    previous_fixes=previous_fixes if previous_fixes else None,
                )
            except Exception as e:
                console.print(f"  [yellow]Debugger failed: {e} — skipping round[/yellow]")
                continue
            previous_fixes.append(fix)
            apply_fix(fix, executor, venv_cmd_fn, safe_write_fn, state)
            verify = executor.run_tests(test_cmd)
            state.exec_history.append(verify)
            if verify.success:
                console.print("  [bold green]Fixed — tests pass again[/bold green]")
                return
        raise PipelineError("Optimization broke tests and could not be repaired")


def apply_fix(fix: DebugFix, executor: ExecutorAgent, venv_cmd_fn, safe_write_fn, state: BuildState) -> None:
    """Apply a debugger fix to the project files on disk."""
    for path, content in fix.file_changes.items():
        safe_write_fn(path, content)
        log("pipeline", f"  patched {path}")
    if fix.new_dependencies:
        install_dependencies(fix.new_dependencies, executor, venv_cmd_fn, state)


def venv_cmd(output_dir: str, cmd: str) -> str:
    """Prefix a command to use the project's venv if it exists."""
    venv = Path(output_dir) / ".venv" / "bin"
    if venv.exists():
        for tool in ("python", "pip", "pytest"):
            if cmd.startswith(tool + " ") or cmd == tool:
                return str(venv / tool) + cmd[len(tool):]
    return cmd


def install_dependencies(
    deps: list[str],
    executor: ExecutorAgent,
    venv_cmd_fn,
    state: BuildState,
) -> None:
    """Install a list of dependency specs safely.

    Strips version specifiers (>=, <=, ~=, !=, [extras]) from dependency
    strings before passing to pip, because the shell safety checker rejects
    metacharacters like > and <.

    pip install gets the latest version; the project's requirements.txt or
    pyproject.toml should carry the real version constraints.
    """
    for dep in deps:
        pkg_name = _strip_version_spec(dep)
        if not pkg_name:
            continue
        result = executor.run_command(venv_cmd_fn(f"pip install {pkg_name}"))
        state.exec_history.append(result)


def _strip_version_spec(dep: str) -> str:
    """Strip version specifiers and extras from a dependency string.

    'pytest>=7.0'     → 'pytest'
    'pydantic[email]' → 'pydantic'
    'uvicorn[standard]>=0.29' → 'uvicorn'
    'requests'        → 'requests'
    ''                → ''
    """
    return re.split(r"[><=!~\[;]", dep)[0].strip()


def read_project_files(output_dir: str) -> dict[str, str]:
    """Read all project files into a dict."""
    out = Path(output_dir)
    files = {}
    for p in out.rglob("*"):
        if p.is_file() and not any(
            skip in str(p) for skip in [".venv", "__pycache__", ".build_state", ".git", "node_modules"]
        ):
            try:
                content = p.read_text(errors="replace")
                if len(content) < 50000:
                    files[str(p.relative_to(out))] = content
            except Exception:
                pass
    return files


def save_state(state: BuildState, output_dir: str) -> None:
    state_dir = Path(output_dir) / ".build_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text(state.model_dump_json(indent=2))


def print_plan(plan: BuildPlan) -> None:
    table = Table(title=f"{plan.project_name} — Build Plan")
    table.add_column("Module", style="cyan")
    table.add_column("Size")
    table.add_column("Deps")
    table.add_column("Files", style="dim")
    for m in plan.modules:
        table.add_row(m.id, m.size.value, ", ".join(m.dependencies) or "—", str(len(m.file_paths)))
    console.print(table)
    console.print(f"  Build order: {plan.build_order}")


def print_final_report(state: BuildState) -> None:
    acc = state.acceptance
    if acc and acc.verdict == AcceptanceVerdict.PASS:
        status, color = "PASS", "green"
    elif acc and acc.verdict.value == "incomplete":
        status, color = "INCOMPLETE", "yellow"
    else:
        status, color = "FAIL", "red"

    report = Table(title="Build Report")
    report.add_column("Metric", style="bold")
    report.add_column("Value")
    report.add_row("Idea", state.idea[:100])
    report.add_row("Modules built", str(len(state.artifacts)))
    report.add_row("Debug rounds", str(state.debug_rounds))
    report.add_row("Optimizations", str(state.optimization_count))
    report.add_row("Acceptance", f"[{color}]{status}[/{color}]")
    report.add_row("Output", state.output_dir)

    # Cost summary
    from build_loop.llm import get_cost_summary
    costs = get_cost_summary()
    if costs["total_calls"] > 0:
        report.add_row("LLM calls", str(costs["total_calls"]))
        report.add_row("Tokens (in/out)", f"{costs['total_input_tokens']:,} / {costs['total_output_tokens']:,}")
        report.add_row("Est. cost", f"${costs['total_cost_usd']:.4f}")
        for model, mc in costs.get("by_model", {}).items():
            short = model.split("-")[1] if "-" in model else model
            report.add_row(f"  {short}", f"{mc['calls']} calls, ${mc['cost_usd']:.4f}")

    if acc:
        report.add_row("Passed", ", ".join(acc.criteria_passed) or "—")
        if acc.criteria_failed:
            report.add_row("Failed", ", ".join(acc.criteria_failed))

    console.print(report)
