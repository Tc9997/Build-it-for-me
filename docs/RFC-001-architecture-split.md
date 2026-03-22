# RFC-001: Architecture Split

**Status:** Draft
**Date:** 2026-03-22

## Summary

Split the current monolithic `build_loop/` package into four logical subsystems with clear boundaries, responsibilities, and interface contracts. This enables independent development, testing, and eventual separate packaging of each subsystem.

## Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        ROUTER                            │
│  CLI entry point, mode selection, resume orchestration   │
│  Owns: main.py, agents/architect.py, modes/__init__.py  │
├───────────────────────┬─────────────────────────────────┤
│  ARCHETYPE COMPILER   │     FREEFORM ITERATION          │
│  ENGINE               │     ENGINE                      │
│                       │                                  │
│  Contract-driven,     │  Prose-driven, broad scope,     │
│  template-first,      │  experimental, LLM-judged       │
│  verifier-backed      │                                  │
│                       │                                  │
│  Owns:                │  Owns:                           │
│   modes/template_     │   modes/freeform.py              │
│     first.py          │   agents/planner.py              │
│   contract.py         │     (FreeformPlannerAgent)       │
│   policy.py           │                                  │
│   environment.py      │                                  │
│   plan_validation.py  │                                  │
│   verifier.py         │                                  │
│   templates/*         │                                  │
│   analysis/*          │                                  │
│   agents/             │                                  │
│     spec_compiler.py  │                                  │
├───────────────────────┴─────────────────────────────────┤
│                  SHARED PLATFORM                         │
│  Agent base, LLM client, executor, safety, schemas,     │
│  build/review/test/debug/optimize pipeline ops           │
│                                                          │
│  Owns:                                                   │
│   agents/base.py, builder.py, reviewer.py,              │
│   integrator.py, executor.py, debugger.py,              │
│   optimizer.py, acceptance.py, researcher.py            │
│   common/pipeline.py                                     │
│   schemas.py, safety.py, llm.py, web.py                 │
│   eval/*                                                 │
└─────────────────────────────────────────────────────────┘
```

## Subsystem Responsibilities

### 1. Shared Platform (`platform/`)

**Responsibility:** Infrastructure that any build mode needs. No policy opinions. No archetype knowledge.

**Promises:**
- Thread-safe LLM client with cost tracking
- Safe command execution (no shell, path traversal prevention)
- Safe URL fetching (no SSRF)
- Build/review loop with deterministic screening
- Test/debug loop with bounded retries
- File write with path safety
- Setup environment (venv, pip)
- Agent base class with error context
- Schemas for artifacts, plans, modules, results
- Eval harness for benchmarking

**Current modules:**

| Module | Status | Notes |
|---|---|---|
| `llm.py` | Clean | No mode-specific logic |
| `safety.py` | Clean | No mode-specific logic |
| `web.py` | Clean | SSRF blocking is universal |
| `schemas.py` | **Mixed** | Contains `ContractState`, `EnvironmentState`, `PolicyState` wrappers that belong to archetype engine |
| `agents/base.py` | Clean | |
| `agents/builder.py` | **Mixed** | Prompt mentions pyproject.toml ownership rules — archetype-specific |
| `agents/reviewer.py` | Clean | |
| `agents/integrator.py` | **Mixed** | Prompt mentions template files, export metadata — archetype-specific context |
| `agents/executor.py` | Clean | |
| `agents/debugger.py` | Clean | |
| `agents/optimizer.py` | Clean | |
| `agents/acceptance.py` | **Mixed** | `require_verification` flag and verifier override logic are archetype-specific |
| `agents/researcher.py` | Clean | `light` flag is mode-aware but implementation is universal |
| `common/pipeline.py` | **Mixed** | `_collect_known_modules` and `_screen_syntax` have archetype awareness. Core build/review/test/debug ops are universal. |
| `eval/*` | Clean | |

### 2. Archetype Compiler Engine (`compiler/`)

**Responsibility:** The productized, narrow build path. Takes a typed contract, resolves a pinned template, plans against the contract, builds with ownership enforcement, verifies with deterministic signals.

**Promises:**
- Only `python_cli` and `fastapi_service`
- Typed contract with schema version
- Deterministic policy from contract + environment
- Pinned, cached, ownership-tracked templates
- Plan validation (goal coverage, build order, same-batch deps)
- Independent verifier as authority (tier 1 + tier 2 + tier 3)
- Signal derivation from archetype, not LLM imagination
- Post-write and post-setup deterministic checks
- Fail-closed on every gate

**Current modules:**

| Module | Status | Notes |
|---|---|---|
| `modes/template_first.py` | Sole owner | Orchestrator |
| `contract.py` | Clean | |
| `policy.py` | Clean | |
| `environment.py` | Clean | |
| `plan_validation.py` | Clean | |
| `verifier.py` | Clean | |
| `templates/*` | Clean | Registry, cache, materialize, ownership, models |
| `analysis/*` | Clean | Exports, contract validation, framework hints, archetype checks, signal derivation, post-write, README validation |
| `agents/spec_compiler.py` | Clean | |
| `agents/planner.py` | **Mixed** | `PlannerAgent` (contract-aware) belongs here. `FreeformPlannerAgent` belongs to freeform engine. |

### 3. Freeform Iteration Engine (`freeform/`)

**Responsibility:** The experimental, broad build path. Prose-driven, no contract, no templates, no verifier. LLM acceptance is the final judge.

**Promises:**
- Any project type
- Best-effort quality
- Labeled experimental everywhere
- LLM verdict can pass (not forced INCOMPLETE)

**Current modules:**

| Module | Status | Notes |
|---|---|---|
| `modes/freeform.py` | Sole owner | Orchestrator |
| `agents/planner.py` | **Mixed** | `FreeformPlannerAgent` belongs here |

### 4. Router (`router/`)

**Responsibility:** CLI entry point, mode selection, resume orchestration. Thin — delegates to engines.

**Promises:**
- `--mode template_first` (default) or `--mode freeform`
- `--resume` from saved state
- Exit code reflects mode-appropriate success criteria
- Lazy imports — broken archetype engine doesn't break freeform

**Current modules:**

| Module | Status | Notes |
|---|---|---|
| `main.py` | Clean | |
| `agents/architect.py` | Clean | Thin router |
| `modes/__init__.py` | Clean | `BuildMode` enum |

## Mixed-Responsibility Modules (Must Split)

### `schemas.py`
- **Platform:** `BuildArtifact`, `BuildPlan`, `ModuleSpec`, `InterfaceContract`, `ReviewResult`, `IntegrationResult`, `ExecResult`, `DebugFix`, `AcceptanceResult`, `BuildState` (core fields)
- **Compiler:** `ContractState`, `EnvironmentState`, `PolicyState` wrappers, `BuildState.contract/environment/policy/verification` fields

**Split plan:** Move typed state wrappers to `compiler/state.py`. Platform `BuildState` becomes a base class without contract/policy fields. Compiler extends it.

### `agents/planner.py`
- **Platform:** None — planning is always mode-specific
- **Compiler:** `PlannerAgent` with contract-aware prompt
- **Freeform:** `FreeformPlannerAgent` with prose prompt

**Split plan:** Move `FreeformPlannerAgent` to `freeform/planner.py`. Move `PlannerAgent` to `compiler/planner.py`. Remove `agents/planner.py`.

### `agents/builder.py`
- **Platform:** `BuilderAgent` class, dependency context, framework hints
- **Compiler:** Prompt rules about pyproject.toml, template ownership

**Split plan:** Keep `BuilderAgent` in platform with a generic prompt. Compiler passes archetype-specific prompt additions via a `prompt_additions: list[str]` parameter.

### `agents/integrator.py`
- **Platform:** `IntegratorAgent` class, basic wiring
- **Compiler:** Template file awareness, export metadata context

**Split plan:** Same as builder — keep base in platform, compiler passes context.

### `agents/acceptance.py`
- **Platform:** `AcceptanceAgent` class, LLM acceptance
- **Compiler:** `require_verification` flag, verifier override logic

**Split plan:** Keep base in platform. Compiler wraps with verifier-authority enforcement.

### `common/pipeline.py`
- **Platform:** `build_all`, `build_and_review`, `write_project`, `setup_environment`, `test_and_debug_loop`, `optimize`, `apply_fix`, helpers
- **Compiler:** `_collect_known_modules`, `_screen_syntax` (unresolved import detection uses plan-aware module paths)

**Split plan:** Extract `_collect_known_modules` and the cross-module import screening into `compiler/screening.py`. Platform pipeline becomes fully mode-agnostic.

## Migration Plan

### Step 1: Extract platform schemas
Move universal schemas to `build_loop/platform/schemas.py`. Keep `ContractState`/`EnvironmentState`/`PolicyState` in current location as re-exports. No behavior change.

### Step 2: Split planner agents
Move `FreeformPlannerAgent` to `build_loop/modes/freeform_planner.py`. Move `PlannerAgent` to `build_loop/compiler/planner.py`. Update imports. Old `agents/planner.py` becomes a re-export shim.

### Step 3: Add prompt_additions to builder/integrator
Add `prompt_additions: list[str] = []` to `BuilderAgent.run()` and `IntegratorAgent.run()`. Template_first orchestrator passes archetype-specific rules. Remove hardcoded rules from builder/integrator prompts. No behavior change — same rules, different injection point.

### Step 4: Extract compiler screening
Move `_collect_known_modules` and the import-aware parts of `_screen_syntax` to `build_loop/compiler/screening.py`. Pipeline's `_screen_syntax` takes an optional callback for cross-module validation. Compiler provides the callback; freeform doesn't.

### Step 5: Move compiler modules
Move `contract.py`, `policy.py`, `environment.py`, `plan_validation.py`, `verifier.py`, `templates/`, `analysis/`, `agents/spec_compiler.py` under `build_loop/compiler/`. Add re-export shims at old locations. No behavior change.

### Step 6: Move freeform modules
Move `modes/freeform.py` and `freeform_planner.py` under `build_loop/freeform/`. Add re-export shim. No behavior change.

### Step 7: Extract acceptance verifier enforcement
Move `require_verification` logic and verifier-override from `AcceptanceAgent` to a compiler-side wrapper. Platform `AcceptanceAgent` becomes purely LLM-based. Compiler wraps it with authority enforcement.

### Step 8: Final cleanup
Remove re-export shims. Update all imports to canonical locations. Update ARCHITECTURE.md. Tag as v0.2.0.

## Dependency Rules

```
router → compiler, freeform, platform
compiler → platform (never freeform)
freeform → platform (never compiler)
platform → nothing (leaf)
```

A broken compiler must not prevent freeform from loading (already enforced via lazy imports in `architect.py`).

## What This RFC Does NOT Change

- No behavior changes
- No new features
- No prompt changes
- No test changes (except import path updates during migration)
- No CLI changes
- The system continues to work exactly as it does today throughout all 8 steps

## Success Criteria

After step 8:
- `pytest -q` passes
- `build-loop "a CLI to convert CSV to JSON"` produces the same output
- `build-loop --mode freeform "scrape wine auctions"` produces the same output
- `--resume` works identically
- Each subsystem can be tested independently: `pytest tests/test_platform/`, `pytest tests/test_compiler/`, etc.
- Import cycle analysis shows no violations of the dependency rules
