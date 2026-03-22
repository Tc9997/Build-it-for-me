"""Base agent class. All agents inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rich.console import Console

from build_loop import llm

console = Console()


class Agent(ABC):
    """Base class for all agents in the build loop."""

    name: str = "base"
    system_prompt: str = "You are a helpful assistant."
    model: str = "claude-sonnet-4-20250514"

    def log(self, msg: str) -> None:
        console.print(f"[bold cyan][{self.name}][/bold cyan] {msg}")

    def call(self, user_msg: str, max_tokens: int = 16384) -> str:
        self.log("thinking...")
        try:
            return llm.call(
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_msg}],
                model=self.model,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"[{self.name}] LLM call failed: {e}") from e

    def call_json(self, user_msg: str, max_tokens: int = 16384) -> dict:
        self.log("thinking...")
        try:
            return llm.call_json(
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_msg}],
                model=self.model,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"[{self.name}] LLM call failed: {e}") from e

    @abstractmethod
    def run(self, *args, **kwargs):
        ...
