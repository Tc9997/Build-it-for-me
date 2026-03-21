"""Optimizer agent: reviews working code and optimizes for performance, resource usage, and robustness."""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import BuildPlan, ExecResult


SYSTEM = """\
You are the Optimizer agent in an automated build system. You receive a working, tested project \
and optimize it. The code already passes tests — your job is to make it better without breaking it.

You MUST respond with a single JSON object:

{
  "optimizations": [
    {
      "file": "relative/path.py",
      "category": "performance" | "memory" | "io" | "concurrency" | "caching" | "error_handling" | "security" | "resource_cleanup",
      "description": "what you changed and why",
      "impact": "high" | "medium" | "low"
    }
  ],
  "file_changes": {"relative/path.py": "FULL new file contents", ...},
  "new_dependencies": ["any new packages needed"],
  "notes": "string — overall summary of optimizations"
}

Look for and fix ALL of the following:

PERFORMANCE:
- O(n²) or worse algorithms that could be O(n) or O(n log n)
- Repeated computation that should be cached (functools.lru_cache, memoization)
- Synchronous I/O that could be async or batched
- N+1 query patterns (loop of API calls / DB queries that should be batched)
- String concatenation in loops (use join or StringIO)
- Unnecessary data copies (list comprehension where generator suffices)

CONCURRENCY:
- Sequential I/O-bound work that should use asyncio, ThreadPoolExecutor, or aiohttp
- Missing connection pooling for HTTP/DB clients
- Blocking calls in async code

CACHING & I/O:
- Repeated file reads / API calls for the same data
- Missing disk caching for expensive API responses
- Unbuffered I/O for large data

RESOURCE MANAGEMENT:
- File handles / connections / sessions not using context managers
- Missing timeouts on HTTP requests, DB connections, subprocess calls
- Missing retry logic with backoff for flaky external calls

ERROR HANDLING:
- Bare except clauses
- Swallowed exceptions that should log or re-raise
- Missing cleanup in error paths

SECURITY:
- User input passed to shell commands without sanitization
- SQL built with string formatting instead of parameterized queries
- Hardcoded secrets that should come from env vars

Rules:
- ONLY optimize things that have real impact. Don't micro-optimize.
- Every changed file must be the COMPLETE file contents, not a diff.
- Do NOT change interfaces, function signatures, or module boundaries.
- Do NOT change test files — if your optimization breaks tests, it's wrong.
- If there's nothing meaningful to optimize, return empty optimizations and file_changes.
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class OptimizerAgent(Agent):
    name = "optimizer"
    system_prompt = SYSTEM

    def run(
        self,
        plan: BuildPlan,
        project_files: dict[str, str],
        test_result: ExecResult | None = None,
    ) -> dict:
        prompt_parts = [
            f"PROJECT: {plan.project_name}",
            f"TECH STACK: {', '.join(plan.tech_stack)}",
            f"\nPROJECT FILES:\n{json.dumps(project_files, indent=2)}",
        ]

        if test_result and test_result.stdout:
            prompt_parts.append(
                f"\nTEST OUTPUT (for reference — tests pass):\n{test_result.stdout[-3000:]}"
            )

        data = self.call_json("\n".join(prompt_parts))

        opt_count = len(data.get("optimizations", []))
        file_count = len(data.get("file_changes", {}))
        if opt_count:
            self.log(f"found {opt_count} optimizations across {file_count} files")
            for opt in data.get("optimizations", []):
                self.log(f"  [{opt.get('impact', '?')}] {opt.get('category', '?')}: {opt.get('description', '')[:80]}")
        else:
            self.log("no meaningful optimizations found")

        return data
