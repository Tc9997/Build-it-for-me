"""Thin wrapper around the Anthropic API with per-call cost tracking."""

from __future__ import annotations

import json
import os
import re
import threading

import anthropic


_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()

# Per-call cost tracking
_cost_lock = threading.Lock()
_cost_log: list[dict] = []


# Approximate cost per 1M tokens by model (input, output)
_COST_PER_M: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}

# Regex to strip markdown code fences (handles ```, ```json, ```python, etc.)
_FENCE_START = re.compile(r"^```\w*\s*\n?", re.MULTILINE)
_FENCE_END = re.compile(r"\n?```\s*$", re.MULTILINE)


def get_client() -> anthropic.Anthropic:
    global _client
    with _client_lock:
        if _client is None:
            _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return _client


def call(
    system: str,
    messages: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> str:
    """Send a message and return the text response."""
    resp = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    )
    _record_cost(model, resp.usage)
    return resp.content[0].text


def call_json(
    system: str,
    messages: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> dict:
    """Send a message and parse the response as JSON."""
    resp = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    )
    _record_cost(model, resp.usage)
    raw = resp.content[0].text

    # Detect truncated output (hit max_tokens)
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(
            f"LLM response truncated at {max_tokens} tokens (stop_reason=max_tokens). "
            f"Output is incomplete and cannot be parsed as JSON. "
            f"Last 100 chars: ...{raw[-100:]}"
        )

    text = _strip_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM response is not valid JSON: {e}. "
            f"First 200 chars: {text[:200]}"
        ) from e


def _strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    Handles: ```json, ```python, ````, and bare ```.
    Robust against partial fences, extra backticks, and missing newlines.
    """
    text = text.strip()
    # Remove leading fence (``` or ```json or ````python etc.)
    text = _FENCE_START.sub("", text, count=1)
    # Remove trailing fence
    text = _FENCE_END.sub("", text, count=1)
    return text.strip()


def _record_cost(model: str, usage) -> None:
    """Record token usage and estimated cost."""
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    rates = _COST_PER_M.get(model, (3.0, 15.0))
    cost = (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000

    with _cost_lock:
        _cost_log.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })


def get_cost_summary() -> dict:
    """Get aggregate cost summary across all calls."""
    with _cost_lock:
        if not _cost_log:
            return {"total_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_cost_usd": 0.0}

        total_input = sum(c["input_tokens"] for c in _cost_log)
        total_output = sum(c["output_tokens"] for c in _cost_log)
        total_cost = sum(c["cost_usd"] for c in _cost_log)

        by_model = {}
        for c in _cost_log:
            m = c["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            by_model[m]["calls"] += 1
            by_model[m]["input_tokens"] += c["input_tokens"]
            by_model[m]["output_tokens"] += c["output_tokens"]
            by_model[m]["cost_usd"] = round(by_model[m]["cost_usd"] + c["cost_usd"], 6)

        return {
            "total_calls": len(_cost_log),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 4),
            "by_model": by_model,
        }


def reset_cost_tracking() -> None:
    """Reset cost tracking (for testing)."""
    with _cost_lock:
        _cost_log.clear()
