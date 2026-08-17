from pathlib import Path

import pytest

import rvai
from rvai.registry import ModelNotFoundError, ModelRegistry

MODELS_DIR = Path(__file__).parents[1] / "models"
PACKAGED_MODELS_DIR = Path(rvai.__file__).parent / "data" / "models"


def test_registry_lists_expected_models() -> None:
    names = [manifest.name for manifest in ModelRegistry(MODELS_DIR).list()]

    assert names == [
        "builtin-gemm-int8",
        "mobilenet-int8",
        "mobilenet-v2-fp32-onnx",
        "qwen-small-int4",
    ]


def test_registry_reports_unknown_model() -> None:
    with pytest.raises(ModelNotFoundError, match="Unknown model 'missing-model'"):
        ModelRegistry(MODELS_DIR).get("missing-model")


def test_packaged_manifests_match_repository_manifests() -> None:
    repository_files = {
        path.name: path.read_bytes() for path in sorted(MODELS_DIR.glob("*.yaml"))
    }
    packaged_files = {
        path.name: path.read_bytes()
        for path in sorted(PACKAGED_MODELS_DIR.glob("*.yaml"))
    }

    assert packaged_files == repository_files
