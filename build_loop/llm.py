"""Thin wrapper around the Anthropic API."""

from __future__ import annotations

import json
import os

import anthropic


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
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
    return resp.content[0].text


def call_json(
    system: str,
    messages: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> dict:
    """Send a message and parse the response as JSON."""
    raw = call(system, messages, model=model, max_tokens=max_tokens, temperature=temperature)
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return json.loads(text)
