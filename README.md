# Build-it-for-me

Multi-agent build system for scaffolding Python CLI tools and FastAPI services from plain English descriptions. Template-first mode produces tested, ownership-tracked projects for two supported archetypes. Freeform mode is experimental and best-effort.

## Two modes

### `template_first` (default — the product)

Narrow, reliable build appliance for two supported archetypes:

- **`python_cli`** — CLI tools, scripts, data pipelines
- **`fastapi_service`** — REST APIs, web services

Pipeline: Research → Contract → Environment → Policy → Template → Plan → Build → Integrate → Write → Setup → Test+Debug → Verify → Optimize → Accept

Features: typed contracts, deterministic policy, commit-pinned templates, ownership manifests, independent verifier as authority.

```bash
build-loop "a CLI to convert CSV to SQL inserts"
build-loop "a REST API for managing book reviews"
```

### `freeform` (experimental)

Broad, best-effort generalist loop for any project type. Useful for exploration and benchmarking. **Not the product promise.**

No contract, no templates, no ownership enforcement, no verifier. LLM-based acceptance only.

```bash
build-loop --mode freeform "scrape wine auctions and recommend based on my taste"
build-loop --mode freeform "find smart devices on my LAN and control via WhatsApp"
```

## Install

```bash
pip install -e ".[test]"
```

## Usage

```bash
# Default (template_first)
build-loop "your idea"

# Explicit mode
build-loop --mode template_first "a CLI tool for X"
build-loop --mode freeform "anything you want to try"

# From file, custom output
build-loop -f idea.txt -o ~/my-project
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for module boundaries, mode separation, and criteria for a future repo split.

## Tests

```bash
python -m pytest tests/ -v
```
