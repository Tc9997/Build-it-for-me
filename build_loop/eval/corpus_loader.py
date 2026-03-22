"""Load eval tasks from the corpus directory."""

from __future__ import annotations

import json
from pathlib import Path

from build_loop.eval.models import EvalTask


def _corpus_dir() -> Path:
    return Path(__file__).parent / "corpus"


def load_all() -> list[EvalTask]:
    """Load all eval tasks from the corpus."""
    tasks = []
    for f in sorted(_corpus_dir().glob("*.json")):
        data = json.loads(f.read_text())
        tasks.append(EvalTask(**data))
    return tasks


def load_by_archetype(archetype: str) -> list[EvalTask]:
    """Load eval tasks for a specific archetype."""
    return [t for t in load_all() if t.archetype == archetype]


def load_by_id(task_id: str) -> EvalTask | None:
    """Load a single eval task by ID."""
    for t in load_all():
        if t.id == task_id:
            return t
    return None
