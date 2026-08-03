import hashlib
import importlib.util
from pathlib import Path

import pytest

from rvai.adapters import OnnxRuntimeAdapter
from rvai.inference import InferenceError, load_onnx_dependencies
from rvai.manifest import ModelManifest


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "onnx"
MODEL_PATH = FIXTURE_DIR / "tiny-classifier.onnx"
IMAGE_PATH = FIXTURE_DIR / "red-image.ppm"
pytestmark = pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("numpy", "onnxruntime", "PIL")
    ),
    reason="ONNX optional dependencies are not installed",
)


def manifest() -> ModelManifest:
    model_bytes = MODEL_PATH.read_bytes()
    return ModelManifest.model_validate(
        {
            "name": "tiny-classifier",
            "display_name": "Tiny Classifier",
            "task": "image_classification",
            "format": "onnx",
            "quantization": "fp32",
            "runtime": "onnxruntime",
            "resources": {
                "min_memory_mb": 1,
                "recommended_threads": "auto",
            },
            "riscv": {"require_rv64": False, "prefer_rvv": False},
            "artifact": {
                "filename": "tiny-classifier.onnx",
                "url": "https://example.com/tiny-classifier.onnx",
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                "size_bytes": len(model_bytes),
            },
            "input": {
                "type": "image",
                "width": 2,
                "height": 2,
                "layout": "nchw",
                "dtype": "float32",
                "color_space": "rgb",
                "resize": "bilinear",
                "normalize": {
                    "scale": 1.0,
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                },
            },
            "output": {
                "type": "classification",
                "top_k": 3,
                "labels": "rgb",
                "scores": "logits",
            },
        }
    )


def test_adapter_executes_real_tiny_onnx_model() -> None:
    result = OnnxRuntimeAdapter().infer(
        manifest(),
        model_path=MODEL_PATH,
        input_path=IMAGE_PATH,
    )

    assert result.status == "success"
    assert result.runtime == "onnxruntime"
    assert result.predictions[0].index == 0
    assert result.predictions[0].label == "rgb:0"
    assert result.execution.provider == "CPUExecutionProvider"
    assert result.execution.latency_ms >= 0


def test_adapter_rejects_runtime_input_shape_mismatch() -> None:
    dependencies = load_onnx_dependencies()

    class Metadata:
        name = "image"
        type = "tensor(float)"
        shape = [1, 3, 4, 4]

    class FakeSession:
        def get_inputs(self):
            return [Metadata()]

        def get_outputs(self):
            return [Metadata()]

    adapter = OnnxRuntimeAdapter(
        dependencies=dependencies,
        session_factory=lambda *args, **kwargs: FakeSession(),
    )

    with pytest.raises(InferenceError, match="does not match"):
        adapter.infer(manifest(), model_path=MODEL_PATH, input_path=IMAGE_PATH)
