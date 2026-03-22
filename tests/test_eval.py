"""Tests for the eval harness: corpus loading, models, reporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_loop.eval.corpus_loader import load_all, load_by_archetype, load_by_id
from build_loop.eval.models import EvalRunResult, EvalSuiteResult, EvalTask
from build_loop.eval.reporter import save_results


class TestCorpusLoader:
    """Corpus must load and contain expected tasks."""

    def test_loads_all_tasks(self):
        tasks = load_all()
        assert len(tasks) >= 10  # 5 python_cli + 5 fastapi_service

    def test_all_tasks_have_required_fields(self):
        for task in load_all():
            assert task.id
            assert task.name
            assert task.archetype in ("python_cli", "fastapi_service")
            assert task.idea
            assert len(task.idea) > 20

    def test_load_by_archetype_python_cli(self):
        tasks = load_by_archetype("python_cli")
        assert len(tasks) >= 5
        assert all(t.archetype == "python_cli" for t in tasks)

    def test_load_by_archetype_fastapi_service(self):
        tasks = load_by_archetype("fastapi_service")
        assert len(tasks) >= 5
        assert all(t.archetype == "fastapi_service" for t in tasks)

    def test_load_by_id(self):
        task = load_by_id("python_cli_01")
        assert task is not None
        assert task.name == "CSV to JSON converter"

    def test_load_by_id_not_found(self):
        assert load_by_id("nonexistent_task") is None

    def test_task_ids_unique(self):
        tasks = load_all()
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

    def test_expected_signals_are_valid(self):
        """Every expected signal must have a type and description."""
        for task in load_all():
            for sig in task.expected_signals:
                assert "type" in sig, f"Signal missing type in {task.id}"
                assert "description" in sig, f"Signal missing description in {task.id}"


class TestEvalModels:
    """Eval result models must serialize cleanly."""

    def test_run_result_defaults(self):
        r = EvalRunResult(task_id="t1", task_name="test", archetype="python_cli", mode="template_first")
        assert not r.passed
        assert r.debug_rounds == 0
        assert r.wall_time_seconds == 0.0

    def test_suite_result_aggregation(self):
        s = EvalSuiteResult(
            mode="template_first",
            total_tasks=3,
            tasks_passed=2,
            tasks_failed=1,
            pass_rate=0.6667,
        )
        assert s.pass_rate == 0.6667
        assert s.total_tasks == 3

    def test_results_serialize_to_json(self):
        r = EvalRunResult(
            task_id="t1", task_name="test", archetype="python_cli",
            mode="template_first", passed=True, wall_time_seconds=42.5,
        )
        j = r.model_dump_json()
        restored = EvalRunResult.model_validate_json(j)
        assert restored.passed
        assert restored.wall_time_seconds == 42.5


class TestEvalReporter:
    """Reporter must write valid JSON."""

    def test_save_results(self, tmp_path):
        suite = EvalSuiteResult(
            mode="template_first", total_tasks=1, tasks_passed=1,
            pass_rate=1.0, results=[
                EvalRunResult(
                    task_id="t1", task_name="test", archetype="python_cli",
                    mode="template_first", passed=True,
                ),
            ],
        )
        output = tmp_path / "results.json"
        save_results([suite], output)

        assert output.exists()
        data = json.loads(output.read_text())
        assert "timestamp" in data
        assert len(data["suites"]) == 1
        assert data["suites"][0]["pass_rate"] == 1.0
