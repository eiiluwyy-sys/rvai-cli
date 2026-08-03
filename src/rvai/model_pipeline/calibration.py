"""Manifest-ordered calibration data preparation for MobileNetV2 P4.3B."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

from pydantic import PositiveInt, model_validator

from rvai.inference.errors import InferenceError
from rvai.inference.preprocess import preprocess_image
from rvai.manifest import ImageInputSpec
from rvai.model_pipeline.dataset import ResolvedDatasetSample, ValidatedDataset
from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.schema import (
    Identifier,
    MobileNetV2P43BPreprocessingConfig,
    Sha256Digest,
    StrictModel,
)


class CalibrationError(ModelPipelineError):
    """Raised when deterministic calibration data cannot be prepared."""


@dataclass(frozen=True)
class ModelPipelineDependencies:
    """Lazily imported optional modules required for model production."""

    numpy: ModuleType
    onnx: ModuleType
    onnxruntime: ModuleType
    quantization: ModuleType
    pillow_image: ModuleType


class MobileNetV2P43BCalibrationSelectionRecord(StrictModel):
    """Deterministic identity and order of selected calibration samples."""

    schema_version: Literal["1.0"] = "1.0"
    manifest_sha256: Sha256Digest
    sample_order: Literal["manifest"]
    sample_count: PositiveInt
    sample_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def count_matches_samples(self) -> "MobileNetV2P43BCalibrationSelectionRecord":
        if self.sample_count != len(self.sample_ids):
            raise ValueError("sample_count must equal the number of sample_ids")
        return self


@dataclass(frozen=True)
class CalibrationSelection:
    """Runtime sample paths paired with deterministic selection evidence."""

    dataset: ValidatedDataset
    samples: tuple[ResolvedDatasetSample, ...]
    record: MobileNetV2P43BCalibrationSelectionRecord


def load_model_pipeline_dependencies() -> ModelPipelineDependencies:
    """Import model-production dependencies only when B2 is requested."""

    modules: dict[str, ModuleType] = {}
    missing: list[str] = []
    for distribution, module_name in (
        ("numpy", "numpy"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("onnxruntime", "onnxruntime.quantization"),
        ("Pillow", "PIL.Image"),
    ):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            if distribution not in missing:
                missing.append(distribution)
    if missing:
        raise CalibrationError(
            "Model-pipeline dependencies are missing ("
            + ", ".join(missing)
            + '); install them with pip install -e ".[model-pipeline]"'
        )
    return ModelPipelineDependencies(
        numpy=modules["numpy"],
        onnx=modules["onnx"],
        onnxruntime=modules["onnxruntime"],
        quantization=modules["onnxruntime.quantization"],
        pillow_image=modules["PIL.Image"],
    )


def select_calibration_samples(
    dataset: ValidatedDataset,
    sample_count: int,
) -> CalibrationSelection:
    """Select an exact manifest-order prefix without repetition or fallback."""

    if dataset.manifest.dataset.purpose != "calibration":
        raise CalibrationError("Calibration selection requires a calibration dataset")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise CalibrationError("Calibration sample_count must be a positive integer")
    if len(dataset.samples) < sample_count:
        raise CalibrationError(
            f"Calibration requires {sample_count} samples, "
            f"but the validated manifest contains {len(dataset.samples)}"
        )
    samples = dataset.samples[:sample_count]
    record = MobileNetV2P43BCalibrationSelectionRecord(
        manifest_sha256=dataset.record.manifest_sha256,
        sample_order="manifest",
        sample_count=sample_count,
        sample_ids=tuple(sample.declaration.id for sample in samples),
    )
    return CalibrationSelection(dataset=dataset, samples=samples, record=record)


class ManifestCalibrationDataReader:
    """Duck-typed ONNX Runtime reader preserving the selected manifest order."""

    def __init__(
        self,
        selection: CalibrationSelection,
        *,
        input_name: str,
        preprocessing: MobileNetV2P43BPreprocessingConfig,
        dependencies: ModelPipelineDependencies | None = None,
    ) -> None:
        if not input_name or input_name != input_name.strip() or "\x00" in input_name:
            raise CalibrationError("ONNX input_name must be a clean non-empty string")
        self.selection = selection
        self.input_name = input_name
        self.preprocessing = preprocessing
        self.dependencies = dependencies or load_model_pipeline_dependencies()
        self._input_spec = _image_input_spec(preprocessing)
        self._index = 0

    def get_next(self) -> dict[str, Any] | None:
        """Return the next preprocessed tensor or the required end sentinel."""

        if self._index >= len(self.selection.samples):
            return None
        sample = self.selection.samples[self._index]
        self._index += 1
        try:
            tensor, _ = preprocess_image(
                sample.resolved_path,
                self._input_spec,
                numpy=self.dependencies.numpy,
                pillow_image=self.dependencies.pillow_image,
            )
        except InferenceError as exc:
            raise CalibrationError(
                f"Cannot preprocess calibration sample {sample.declaration.id}: {exc}"
            ) from exc
        return {self.input_name: tensor}

    def rewind(self) -> None:
        """Restart deterministic iteration at the first selected sample."""

        self._index = 0


def _image_input_spec(
    preprocessing: MobileNetV2P43BPreprocessingConfig,
) -> ImageInputSpec:
    return ImageInputSpec.model_validate(
        {
            "type": preprocessing.media_type,
            "width": preprocessing.width,
            "height": preprocessing.height,
            "layout": preprocessing.layout,
            "dtype": preprocessing.dtype,
            "color_space": preprocessing.color_space,
            "resize": preprocessing.resize,
            "resize_shorter": preprocessing.resize_shorter,
            "crop": preprocessing.crop,
            "normalize": preprocessing.normalize.model_dump(mode="python"),
        }
    )
