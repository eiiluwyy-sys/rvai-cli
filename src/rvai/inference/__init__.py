"""Optional native ONNX image inference APIs."""

from rvai.inference.dependencies import OnnxDependencies, load_onnx_dependencies
from rvai.inference.errors import (
    InferenceDependencyError,
    InferenceError,
    InferenceInputError,
    InferenceOutputError,
)
from rvai.inference.labels import classification_label, load_label_catalog
from rvai.inference.postprocess import classification_top_k
from rvai.inference.preprocess import preprocess_image
from rvai.inference.schema import (
    ClassificationPrediction,
    ExecutionInfo,
    InferenceResult,
    InputInfo,
)

__all__ = [
    "ClassificationPrediction",
    "ExecutionInfo",
    "InferenceDependencyError",
    "InferenceError",
    "InferenceInputError",
    "InferenceOutputError",
    "InferenceResult",
    "InputInfo",
    "OnnxDependencies",
    "classification_top_k",
    "classification_label",
    "load_onnx_dependencies",
    "load_label_catalog",
    "preprocess_image",
]
