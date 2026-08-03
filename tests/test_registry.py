from pathlib import Path

import pytest

from rvai.registry import ModelNotFoundError, ModelRegistry


MODELS_DIR = Path(__file__).parents[1] / "models"


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
