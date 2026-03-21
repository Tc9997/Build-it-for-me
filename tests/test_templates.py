"""Tests for template pinning, ownership enforcement, and archetype rejection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from build_loop.contract import BuildContract
from build_loop.templates.materialize import MaterializationError, materialize
from build_loop.templates.models import Archetype, FileOwner, OwnershipManifest
from build_loop.templates.ownership import OwnershipViolationError, check_write_allowed
from build_loop.templates.registry import (
    RegistryError,
    _compute_content_hash,
    _fixtures_dir,
    resolve,
    verify_commit,
)


# =========================================================================
# True pinning: checked-in hashes, not live recomputation
# =========================================================================

class TestTemplatePinning:
    """Pins must be immutable checked-in hashes, not live recomputation."""

    def test_registry_loads_with_valid_pins(self):
        """Registry should load successfully with matching pins."""
        # This tests the real import-time behavior
        entry = resolve("python_cli")
        assert entry.pinned_commit  # Non-empty hash
        assert len(entry.pinned_commit) == 64  # SHA-256 hex

    def test_verify_commit_matches_pinned_hash(self):
        """verify_commit should pass for untampered fixtures."""
        entry = resolve("python_cli")
        assert verify_commit(entry)

    def test_tampered_fixture_fails_verification(self, tmp_path):
        """If fixture content changes, verification must fail."""
        entry = resolve("python_cli")
        # Copy fixture to tmp and tamper
        tampered = tmp_path / "tampered"
        shutil.copytree(entry.source_path, tampered)
        (tampered / "TAMPERED.txt").write_text("injected file")

        # Create a fake entry pointing to tampered dir
        from build_loop.templates.models import RegistryEntry
        fake_entry = RegistryEntry(
            template_id="python_cli_v1",
            archetype=Archetype.PYTHON_CLI,
            source_path=str(tampered),
            pinned_commit=entry.pinned_commit,  # Original pin
        )
        assert not verify_commit(fake_entry)

    def test_pinned_hashes_file_exists(self):
        """The checked-in hashes file must exist."""
        from build_loop.templates.registry import _pinned_hashes_path
        assert _pinned_hashes_path().exists()

    def test_pinned_hashes_are_not_empty(self):
        from build_loop.templates.registry import _pinned_hashes_path
        hashes = json.loads(_pinned_hashes_path().read_text())
        assert "python_cli_v1" in hashes
        assert "fastapi_service_v1" in hashes
        assert all(len(h) == 64 for h in hashes.values())


# =========================================================================
# Ownership fails closed on missing/malformed entries
# =========================================================================

class TestOwnershipFailsClosed:
    """Template files missing from ownership.json must fail materialization."""

    def test_missing_ownership_entry_fails(self, tmp_path):
        """A template file not listed in ownership.json causes hard failure."""
        # Create a minimal template with incomplete ownership.json
        template = tmp_path / "template"
        template.mkdir()
        (template / "ownership.json").write_text('{"existing.py": "template_locked"}')
        (template / "existing.py").write_text("pass")
        (template / "unlisted.py").write_text("pass")  # Not in ownership.json

        with pytest.raises(MaterializationError, match="no ownership entry"):
            materialize(
                cached_template=template,
                output_dir=tmp_path / "out",
                project_name="test",
                summary="test",
                template_id="test",
                pinned_commit="abc",
                contract_hash="def",
            )

    def test_invalid_ownership_value_fails(self, tmp_path):
        """An invalid ownership string causes hard failure."""
        template = tmp_path / "template"
        template.mkdir()
        (template / "ownership.json").write_text('{"file.py": "bogus_value"}')
        (template / "file.py").write_text("pass")

        with pytest.raises(MaterializationError, match="invalid ownership"):
            materialize(
                cached_template=template,
                output_dir=tmp_path / "out",
                project_name="test",
                summary="test",
                template_id="test",
                pinned_commit="abc",
                contract_hash="def",
            )

    def test_valid_ownership_succeeds(self, tmp_path):
        """A complete, valid ownership.json succeeds."""
        template = tmp_path / "template"
        template.mkdir()
        (template / "ownership.json").write_text(
            '{"file.py": "template_locked", "slot.py": "builder_owned"}'
        )
        (template / "file.py").write_text("locked content")
        (template / "slot.py").write_text("slot content")

        manifest = materialize(
            cached_template=template,
            output_dir=tmp_path / "out",
            project_name="test",
            summary="test",
            template_id="test",
            pinned_commit="abc",
            contract_hash="def",
        )
        assert manifest.files["file.py"] == FileOwner.TEMPLATE_LOCKED
        assert manifest.files["slot.py"] == FileOwner.BUILDER_OWNED

    def test_ownership_enforcement_blocks_locked(self):
        """Writing to a template_locked path must raise."""
        manifest = OwnershipManifest(
            template_id="t", pinned_commit="p", contract_hash="c",
            files={"locked.py": FileOwner.TEMPLATE_LOCKED},
        )
        with pytest.raises(OwnershipViolationError):
            check_write_allowed(manifest, "locked.py")

    def test_ownership_allows_builder_owned(self):
        """Writing to a builder_owned path must succeed."""
        manifest = OwnershipManifest(
            template_id="t", pinned_commit="p", contract_hash="c",
            files={"slot.py": FileOwner.BUILDER_OWNED},
        )
        check_write_allowed(manifest, "slot.py")  # No exception

    def test_ownership_allows_new_generated_files(self):
        """Writing a new file (not in manifest) must succeed — it's GENERATED."""
        manifest = OwnershipManifest(
            template_id="t", pinned_commit="p", contract_hash="c",
            files={},
        )
        check_write_allowed(manifest, "new_file.py")  # No exception


# =========================================================================
# Unsupported archetype rejection
# =========================================================================

class TestArchetypeRejection:
    """template_first must reject unsupported archetypes."""

    def test_unsupported_archetype_in_contract(self):
        """A contract with archetype='unsupported' is valid schema but rejected by pipeline."""
        c = BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            archetype="unsupported",
        )
        assert c.archetype == "unsupported"

    def test_registry_rejects_unsupported(self):
        with pytest.raises(RegistryError, match="Unsupported"):
            resolve("unsupported")

    def test_registry_rejects_unknown_string(self):
        with pytest.raises(RegistryError, match="Unsupported"):
            resolve("mobile_app")

    def test_registry_accepts_python_cli(self):
        entry = resolve("python_cli")
        assert entry.archetype == Archetype.PYTHON_CLI

    def test_registry_accepts_fastapi_service(self):
        entry = resolve("fastapi_service")
        assert entry.archetype == Archetype.FASTAPI_SERVICE

    def test_template_first_rejects_unsupported_contract(self):
        """The orchestrator must stop if archetype is 'unsupported'."""
        from build_loop.modes.template_first import TemplateFirstOrchestrator
        from build_loop.common.pipeline import PipelineError
        from build_loop.schemas import ResearchReport

        orch = TemplateFirstOrchestrator(output_dir="/tmp/test-reject")
        orch.researcher.run = MagicMock(return_value=ResearchReport(
            feasibility="test", recommended_stack=[],
        ))
        orch.spec_compiler.run = MagicMock(return_value=BuildContract(
            project_name="test", summary="test",
            goals=["test"], acceptance_criteria=["test"],
            archetype="unsupported",
        ))

        # Should stop at the archetype gate, not reach template resolution
        orch._resolve_and_materialize_template = MagicMock()
        orch.run("build me a mobile app")
        orch._resolve_and_materialize_template.assert_not_called()


# =========================================================================
# Freeform uses a different planner
# =========================================================================

class TestFreeformPlanner:
    """Freeform mode must use FreeformPlannerAgent, not the contract-first one."""

    def test_freeform_uses_freeform_planner(self):
        from build_loop.modes.freeform import FreeformOrchestrator
        from build_loop.agents.planner import FreeformPlannerAgent
        orch = FreeformOrchestrator(output_dir="/tmp/test-freeform")
        assert isinstance(orch.planner, FreeformPlannerAgent)

    def test_template_first_uses_contract_planner(self):
        from build_loop.modes.template_first import TemplateFirstOrchestrator
        from build_loop.agents.planner import PlannerAgent, FreeformPlannerAgent
        orch = TemplateFirstOrchestrator(output_dir="/tmp/test-tf")
        assert isinstance(orch.planner, PlannerAgent)
        assert not isinstance(orch.planner, FreeformPlannerAgent)
