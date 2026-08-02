from pathlib import Path

import pytest

from rvai.planner import RunPlanner
from rvai.registry import ModelRegistry


MODELS_DIR = Path(__file__).parents[1] / "models"


def test_planner_builds_structured_dry_run() -> None:
    manifest = ModelRegistry(MODELS_DIR).get("builtin-gemm-int8")

    plan = RunPlanner().plan(manifest)

    assert plan.model == "builtin-gemm-int8"
    assert plan.runtime == "builtin"
    assert plan.target == "native"
    assert plan.task == "benchmark"
    assert plan.resources["min_memory_mb"] == 128
    assert plan.requires_model_file is False
    assert plan.dry_run is True


def test_planner_rejects_execution_in_v01() -> None:
    manifest = ModelRegistry(MODELS_DIR).get("builtin-gemm-int8")

    with pytest.raises(ValueError, match="only supports dry-run"):
        RunPlanner().plan(manifest, dry_run=False)
