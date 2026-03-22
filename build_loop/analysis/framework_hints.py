"""Static framework version guidance for builders.

Maps tech stack entries to deterministic instruction blocks that prevent
builders from using outdated patterns. No LLM — just pattern rules.
"""

from __future__ import annotations


# Maps a tech stack keyword (case-insensitive) to guidance text
_FRAMEWORK_HINTS: dict[str, str] = {
    "pydantic": (
        "PYDANTIC V2 RULES (CRITICAL — violations will fail review):\n"
        "- Use `from pydantic import BaseModel, Field, model_validator, field_validator`\n"
        "- Do NOT use `@validator` — use `@field_validator` instead\n"
        "- Do NOT use `@root_validator` — use `@model_validator(mode='after')` instead\n"
        "- Do NOT use `.dict()` — use `.model_dump()` instead\n"
        "- Do NOT use `.json()` — use `.model_dump_json()` instead\n"
        "- Do NOT use `.parse_obj()` — use `.model_validate()` instead\n"
        "- Do NOT use `.parse_raw()` — use `.model_validate_json()` instead\n"
        "- Do NOT use `class Config:` inside models — use `model_config = {...}` instead\n"
        "- Do NOT use `from pydantic import GenericModel` — use `class MyModel(BaseModel, Generic[T])` instead\n"
        "- For `extra='forbid'`: use `model_config = {'extra': 'forbid'}`\n"
        "- For discriminated unions: use `Annotated[Union[...], Field(discriminator='type')]`"
    ),
    "fastapi": (
        "FASTAPI RULES:\n"
        "- Use `from fastapi import FastAPI, APIRouter, Depends, HTTPException`\n"
        "- Use Pydantic v2 models for request/response schemas (see Pydantic rules above)\n"
        "- Use `app.include_router(router)` for route organization\n"
        "- Use `@app.get`, `@router.post`, etc. — not `@app.route`\n"
        "- For async endpoints: `async def endpoint()` is preferred"
    ),
    "sqlalchemy": (
        "SQLALCHEMY 2.0 RULES:\n"
        "- Use `from sqlalchemy import create_engine, select, insert`\n"
        "- Use `from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column`\n"
        "- Do NOT use `declarative_base()` — use `class Base(DeclarativeBase): pass`\n"
        "- Do NOT use `Column()` — use `mapped_column()` with `Mapped[type]` annotation\n"
        "- Use `session.execute(select(Model))` — not `session.query(Model)`"
    ),
    "pytest": (
        "PYTEST RULES:\n"
        "- Use `import pytest` and `pytest.raises`, `pytest.fixture`\n"
        "- Do NOT use `unittest.TestCase` — use plain functions or classes\n"
        "- Use `tmp_path` fixture for temporary files, not `tempfile.mkdtemp()`"
    ),
}


def get_framework_hints(tech_stack: list[str]) -> str:
    """Get framework guidance blocks relevant to the tech stack.

    Returns a single string with all applicable hints, or empty string
    if no hints match.
    """
    hints = []
    seen = set()
    for tech in tech_stack:
        tech_lower = tech.lower()
        for keyword, hint in _FRAMEWORK_HINTS.items():
            if keyword in tech_lower and keyword not in seen:
                hints.append(hint)
                seen.add(keyword)
    return "\n\n".join(hints)
