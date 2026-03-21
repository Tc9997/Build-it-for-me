"""Template materialization: instantiate a cached template into a project directory.

Copies template files, applies placeholder substitution, writes the
ownership manifest. Returns the manifest for the orchestrator.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from build_loop.templates.models import FileOwner, OwnershipManifest


class MaterializationError(Exception):
    """Raised for materialization failures."""


MANIFEST_FILENAME = ".ownership.json"


def materialize(
    cached_template: Path,
    output_dir: Path,
    project_name: str,
    summary: str,
    template_id: str,
    pinned_commit: str,
    contract_hash: str,
) -> OwnershipManifest:
    """Instantiate a cached template into the output directory.

    1. Reads the template's ownership.json
    2. Copies files with placeholder substitution
    3. Writes the ownership manifest into the output project
    4. Returns the manifest

    Placeholders: {{project_name}}, {{summary}}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read template ownership declaration
    ownership_path = cached_template / "ownership.json"
    if not ownership_path.exists():
        raise MaterializationError(
            f"Template missing ownership.json: {cached_template}"
        )

    raw_ownership = json.loads(ownership_path.read_text())

    # Build substitution map
    subs = {
        "{{project_name}}": project_name,
        "{{summary}}": summary,
    }

    # Copy and substitute files
    files_map: dict[str, FileOwner] = {}
    for src_path in sorted(cached_template.rglob("*")):
        if not src_path.is_file():
            continue
        rel = str(src_path.relative_to(cached_template))

        # Skip the ownership.json itself — we write our own manifest
        if rel == "ownership.json":
            continue

        # Determine ownership
        owner_str = raw_ownership.get(rel, "generated")
        try:
            owner = FileOwner(owner_str)
        except ValueError:
            owner = FileOwner.GENERATED

        files_map[rel] = owner

        # Copy with substitution
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            content = src_path.read_text()
            for placeholder, value in subs.items():
                content = content.replace(placeholder, value)
            dst.write_text(content)
        except UnicodeDecodeError:
            # Binary file — copy as-is
            shutil.copy2(src_path, dst)

    # Build and write manifest
    manifest = OwnershipManifest(
        template_id=template_id,
        pinned_commit=pinned_commit,
        contract_hash=contract_hash,
        files=files_map,
    )

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    return manifest
