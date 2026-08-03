"""Native ONNX Runtime Adapter for batch-one FP32 image classification."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from rvai.inference import (
    ExecutionInfo,
    InferenceError,
    InferenceResult,
    OnnxDependencies,
    classification_top_k,
    load_onnx_dependencies,
    preprocess_image,
)
from rvai.manifest import ModelManifest


class OnnxRuntimeAdapter:
    """Consume a Resolver-verified ONNX file without download responsibilities."""

    def __init__(
        self,
        *,
        dependencies: OnnxDependencies | None = None,
        session_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.dependencies = dependencies or load_onnx_dependencies()
        self.session_factory = (
            session_factory or self.dependencies.onnxruntime.InferenceSession
        )
        self.clock = clock

    @property
    def name(self) -> str:
        return "onnxruntime"

    @classmethod
    def supports(cls, manifest: ModelManifest) -> bool:
        """Report support without importing or initializing ONNX Runtime."""

        return (
            manifest.runtime == "onnxruntime"
            and manifest.format == "onnx"
            and manifest.task == "image_classification"
            and manifest.quantization == "fp32"
            and manifest.input is not None
            and manifest.output is not None
        )

    def infer(
        self,
        manifest: ModelManifest,
        *,
        model_path: Path,
        input_path: Path,
    ) -> InferenceResult:
        """Execute one native CPU inference and return validated Top-K JSON data."""

        if not self.supports(manifest):
            raise InferenceError(
                f"OnnxRuntimeAdapter does not support model {manifest.name}; "
                "expected FP32 image classification with input/output processing"
            )

        tensor, input_info = preprocess_image(
            input_path,
            manifest.input,
            numpy=self.dependencies.numpy,
            pillow_image=self.dependencies.pillow_image,
        )
        try:
            session = self.session_factory(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or len(outputs) < 1:
                raise InferenceError(
                    "ONNX model must expose exactly one input and at least one output"
                )
            self._validate_input(inputs[0], tensor.shape)
            started = self.clock()
            values = session.run([outputs[0].name], {inputs[0].name: tensor})
            latency_ms = (self.clock() - started) * 1000.0
            providers = session.get_providers()
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"ONNX Runtime inference failed: {exc}") from exc

        if not values:
            raise InferenceError("ONNX Runtime returned no output values")
        predictions = classification_top_k(
            values[0],
            manifest.output,
            numpy=self.dependencies.numpy,
        )
        provider = providers[0] if providers else "CPUExecutionProvider"
        return InferenceResult(
            model=manifest.name,
            input=input_info,
            predictions=predictions,
            execution=ExecutionInfo(
                provider=provider,
                runtime_version=str(
                    getattr(self.dependencies.onnxruntime, "__version__", "unknown")
                ),
                latency_ms=latency_ms,
            ),
        )

    @staticmethod
    def _validate_input(input_metadata: Any, shape: tuple[int, ...]) -> None:
        input_type = getattr(input_metadata, "type", None)
        if input_type != "tensor(float)":
            raise InferenceError(
                f"ONNX input must be tensor(float), got {input_type}"
            )
        expected = getattr(input_metadata, "shape", None)
        if not isinstance(expected, (list, tuple)) or len(expected) != len(shape):
            raise InferenceError(
                f"ONNX input rank does not match prepared tensor shape {shape}"
            )
        for declared, actual in zip(expected, shape):
            if isinstance(declared, int) and declared > 0 and declared != actual:
                raise InferenceError(
                    f"ONNX input shape {expected} does not match prepared tensor {shape}"
                )
