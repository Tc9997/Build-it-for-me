# Architecture: Two Modes in One Repo

## Why two modes?

This repo contains two distinct build pipelines:

- **`template_first`**: The productized path. Narrow scope (python_cli, fastapi_service only), typed contracts, deterministic policy, pinned templates, ownership manifests, independent verification. Reliability-focused.

- **`freeform`**: The experimental path. Broad scope (any project type), prose-driven planning, LLM-based acceptance, no contract or template guarantees. Useful for exploration, research, and benchmarking.

These are not blended. They share common build/review/test machinery but have separate orchestration, separate entry points, and separate quality guarantees.

## Why one repo?

Right now both modes share enough infrastructure that splitting would create maintenance burden:
- Agent implementations (builder, reviewer, integrator, debugger, optimizer)
- Executor (command safety, service smoke testing)
- Safety layer (path traversal, command injection)
- Schema definitions

## When to split into two repos

Split when any of these become true:
1. The shared agent implementations diverge significantly between modes
2. Template-first needs its own release cadence separate from freeform
3. Freeform becomes unmaintained or a security liability
4. The shared code shrinks to less than ~30% of the total codebase

## Module boundaries

```
build_loop/
├── modes/
│   ├── __init__.py          # BuildMode enum
│   ├── template_first.py    # Productized orchestrator
│   └── freeform.py          # Experimental orchestrator
├── common/
│   └── pipeline.py          # Shared: build/review, write, setup, test, debug, optimize
├── agents/
│   ├── architect.py         # Thin router → mode orchestrators
│   ├── base.py              # Base LLM agent
│   ├── builder.py           # Shared
│   ├── reviewer.py          # Shared
│   ├── integrator.py        # Shared
│   ├── executor.py          # Shared (command safety, smoke testing)
│   ├── debugger.py          # Shared
│   ├── optimizer.py         # Shared
│   ├── acceptance.py        # Shared (verifier-aware)
│   ├── researcher.py        # Shared (web search, GitHub)
│   ├── spec_compiler.py     # template_first only
│   └── planner.py           # Shared (prompt differs by mode)
├── templates/               # template_first only
│   ├── registry.py
│   ├── cache.py
│   ├── materialize.py
│   ├── ownership.py
│   └── models.py
├── contract.py              # template_first only
├── environment.py           # template_first only
├── policy.py                # template_first only
├── plan_validation.py       # template_first only
├── verifier.py              # template_first only
├── safety.py                # Shared
├── schemas.py               # Shared
└── main.py                  # CLI with --mode flag
```

## Rules

- `template_first` is the default. CLI routes there unless `--mode freeform` is explicit.
- `template_first` rejects unsupported archetypes. No silent fallback to freeform.
- `freeform` is labeled experimental in every user-facing surface.
- Verifier authority is enforced in `template_first`. Freeform has no verifier.
- Shared code in `common/` must not import from `modes/` or `templates/`.
