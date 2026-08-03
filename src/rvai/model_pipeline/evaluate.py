"""Deterministic FP32 and INT8 evaluation for MobileNetV2 P4.3B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from rvai.inference.errors import InferenceError
from rvai.inference.preprocess import preprocess_image
from rvai.model_pipeline.calibration import (
    ModelPipelineDependencies,
    _image_input_spec,
    load_model_pipeline_dependencies,
)
from rvai.model_pipeline.dataset import ResolvedDatasetSample, ValidatedDataset
from rvai.model_pipeline.errors import ModelPipelineError, PipelineIOError
from rvai.model_pipeline.inspect import MobileNetV2P43BSourceInspectionRecord
from rvai.model_pipeline.inspect import _batch_dimensions_compatible
from rvai.model_pipeline.io import sha256_canonical_json, sha256_file
from rvai.model_pipeline.quantize import MobileNetV2P43BQuantizationRecord
from rvai.model_pipeline.schema import (
    Description,
    Identifier,
    MobileNetV2P43BPipelineConfig,
    PlainFilename,
    Sha256Digest,
    StrictModel,
)


class EvaluationError(ModelPipelineError):
    """Raised when a valid deterministic evaluation cannot be performed."""


Ratio = float
FailureCode = Literal[
    "preprocessing_failed",
    "inference_failed",
    "invalid_output_shape",
    "non_finite_output",
]


class MobileNetV2P43BEvaluationSelectionRecord(StrictModel):
    """Manifest-order evaluation selection without host paths."""

    schema_version: Literal["1.0"] = "1.0"
    manifest_sha256: Sha256Digest
    sample_order: Literal["manifest"]
    requested_sample_count: PositiveInt
    actual_sample_count: PositiveInt
    used_complete_available_set: bool
    sample_ids: tuple[Identifier, ...]

    @field_validator("sample_ids", mode="before")
    @classmethod
    def sample_id_list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def counts_match_samples(self) -> "MobileNetV2P43BEvaluationSelectionRecord":
        if self.actual_sample_count != len(self.sample_ids):
            raise ValueError("actual_sample_count must equal the number of sample_ids")
        if self.actual_sample_count > self.requested_sample_count:
            raise ValueError("actual_sample_count must not exceed requested_sample_count")
        if self.used_complete_available_set != (
            self.actual_sample_count < self.requested_sample_count
        ):
            raise ValueError(
                "used_complete_available_set must identify a short complete dataset"
            )
        return self


@dataclass(frozen=True)
class EvaluationSelection:
    """Runtime sample paths paired with deterministic selection evidence."""

    dataset: ValidatedDataset
    samples: tuple[ResolvedDatasetSample, ...]
    record: MobileNetV2P43BEvaluationSelectionRecord


class MobileNetV2P43BEvaluationArtifact(StrictModel):
    """Identity and role of one model evaluated by B3."""

    role: Literal["fp32", "int8"]
    filename: PlainFilename
    size_bytes: PositiveInt
    sha256: Sha256Digest


class MobileNetV2P43BSampleEvaluation(StrictModel):
    """Portable prediction outcome for one manifest-order sample."""

    sample_id: Identifier
    label: NonNegativeInt
    succeeded: bool
    finite_output: bool
    top1_class: NonNegativeInt | None
    top5_classes: tuple[NonNegativeInt, ...] | None
    top1_correct: bool
    top5_correct: bool
    failure: FailureCode | None

    @field_validator("top5_classes", mode="before")
    @classmethod
    def top5_list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_outcome(self) -> "MobileNetV2P43BSampleEvaluation":
        if self.succeeded:
            if not self.finite_output or self.failure is not None:
                raise ValueError("successful evaluations require a finite output")
            if self.top1_class is None or self.top5_classes is None:
                raise ValueError("successful evaluations require Top-1 and Top-5 data")
            if len(self.top5_classes) != 5 or len(set(self.top5_classes)) != 5:
                raise ValueError("top5_classes must contain five unique classes")
            if self.top1_class != self.top5_classes[0]:
                raise ValueError("top1_class must be the first Top-5 class")
            if self.top1_correct != (self.top1_class == self.label):
                raise ValueError("top1_correct does not match the prediction")
            if self.top5_correct != (self.label in self.top5_classes):
                raise ValueError("top5_correct does not match the prediction")
        elif (
            self.top1_class is not None
            or self.top5_classes is not None
            or self.top1_correct
            or self.top5_correct
            or self.failure is None
        ):
            raise ValueError("failed evaluations must contain only failure evidence")
        return self


class MobileNetV2P43BEvaluationSummary(StrictModel):
    """Accuracy and validity summary over the complete selected set."""

    sample_count: PositiveInt
    successful_inferences: NonNegativeInt
    inference_failures: NonNegativeInt
    non_finite_outputs: NonNegativeInt
    top1_correct: NonNegativeInt
    top5_correct: NonNegativeInt
    top1_accuracy_ratio: Ratio = Field(ge=0.0, le=1.0)
    top5_accuracy_ratio: Ratio = Field(ge=0.0, le=1.0)
    all_outputs_finite: bool

    @model_validator(mode="after")
    def validate_counts(self) -> "MobileNetV2P43BEvaluationSummary":
        if self.successful_inferences + self.inference_failures != self.sample_count:
            raise ValueError("success and failure counts must cover every sample")
        if self.top1_correct > self.successful_inferences:
            raise ValueError("Top-1 correct count exceeds successful inferences")
        if self.top5_correct > self.successful_inferences:
            raise ValueError("Top-5 correct count exceeds successful inferences")
        if self.top1_correct > self.top5_correct:
            raise ValueError("Top-1 correct count must not exceed Top-5")
        if self.non_finite_outputs > self.inference_failures:
            raise ValueError("non-finite outputs must be included in failures")
        if self.top1_accuracy_ratio != self.top1_correct / self.sample_count:
            raise ValueError("Top-1 accuracy does not match its count")
        if self.top5_accuracy_ratio != self.top5_correct / self.sample_count:
            raise ValueError("Top-5 accuracy does not match its count")
        if self.all_outputs_finite != (
            self.inference_failures == 0 and self.non_finite_outputs == 0
        ):
            raise ValueError("all_outputs_finite does not match failure counts")
        return self


class MobileNetV2P43BEvaluationRecord(StrictModel):
    """Deterministic evaluation evidence for one FP32 or INT8 artifact."""

    schema_version: Literal["1.0"] = "1.0"
    model: MobileNetV2P43BEvaluationArtifact
    pipeline_config_sha256: Sha256Digest
    evaluation_manifest_sha256: Sha256Digest
    preprocessing_contract: Literal["mobilenet-v2-imagenet-v1"]
    execution_provider: Literal["CPUExecutionProvider"]
    onnxruntime_version: Description
    selection: MobileNetV2P43BEvaluationSelectionRecord
    samples: tuple[MobileNetV2P43BSampleEvaluation, ...]
    summary: MobileNetV2P43BEvaluationSummary

    @field_validator("samples", mode="before")
    @classmethod
    def sample_list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_record(self) -> "MobileNetV2P43BEvaluationRecord":
        sample_ids = tuple(sample.sample_id for sample in self.samples)
        if sample_ids != self.selection.sample_ids:
            raise ValueError("evaluation samples do not match selection order")
        if self.summary.sample_count != len(self.samples):
            raise ValueError("evaluation summary does not cover every sample")
        if self.evaluation_manifest_sha256 != self.selection.manifest_sha256:
            raise ValueError("evaluation manifest identities do not match")
        successful = sum(sample.succeeded for sample in self.samples)
        failures = len(self.samples) - successful
        non_finite = sum(
            sample.failure == "non_finite_output" for sample in self.samples
        )
        top1_correct = sum(sample.top1_correct for sample in self.samples)
        top5_correct = sum(sample.top5_correct for sample in self.samples)
        if (
            self.summary.successful_inferences != successful
            or self.summary.inference_failures != failures
            or self.summary.non_finite_outputs != non_finite
            or self.summary.top1_correct != top1_correct
            or self.summary.top5_correct != top5_correct
        ):
            raise ValueError("evaluation summary does not match sample outcomes")
        return self


def select_evaluation_samples(
    dataset: ValidatedDataset,
    requested_sample_count: int,
    *,
    allow_complete_dataset_if_fewer: bool = False,
) -> EvaluationSelection:
    """Select a manifest-order prefix, with the documented production fallback."""

    if dataset.manifest.dataset.purpose != "evaluation":
        raise EvaluationError("Evaluation selection requires an evaluation dataset")
    if (
        isinstance(requested_sample_count, bool)
        or not isinstance(requested_sample_count, int)
        or requested_sample_count <= 0
    ):
        raise EvaluationError("Evaluation sample count must be a positive integer")
    if dataset.record.manifest_sha256 != sha256_canonical_json(dataset.manifest):
        raise EvaluationError("Validated evaluation dataset record is inconsistent")
    available = len(dataset.samples)
    if available < requested_sample_count and not allow_complete_dataset_if_fewer:
        raise EvaluationError(
            f"Evaluation requires {requested_sample_count} samples, "
            f"but the validated manifest contains {available}"
        )
    actual = min(available, requested_sample_count)
    samples = dataset.samples[:actual]
    record = MobileNetV2P43BEvaluationSelectionRecord(
        manifest_sha256=dataset.record.manifest_sha256,
        sample_order="manifest",
        requested_sample_count=requested_sample_count,
        actual_sample_count=actual,
        used_complete_available_set=actual < requested_sample_count,
        sample_ids=tuple(sample.declaration.id for sample in samples),
    )
    return EvaluationSelection(dataset=dataset, samples=samples, record=record)


def fp32_evaluation_artifact(
    inspection: MobileNetV2P43BSourceInspectionRecord,
) -> MobileNetV2P43BEvaluationArtifact:
    """Build the FP32 evaluation identity from verified inspection evidence."""

    return MobileNetV2P43BEvaluationArtifact(
        role="fp32",
        filename=inspection.model.filename,
        size_bytes=inspection.model.size_bytes,
        sha256=inspection.model.sha256,
    )


def int8_evaluation_artifact(
    quantization: MobileNetV2P43BQuantizationRecord,
) -> MobileNetV2P43BEvaluationArtifact:
    """Build the INT8 evaluation identity from verified quantization evidence."""

    return MobileNetV2P43BEvaluationArtifact(
        role="int8",
        filename=quantization.artifact.filename,
        size_bytes=quantization.artifact.size_bytes,
        sha256=quantization.artifact.sha256,
    )


def evaluate_model(
    model_path: Path | str,
    *,
    artifact: MobileNetV2P43BEvaluationArtifact,
    pipeline: MobileNetV2P43BPipelineConfig,
    evaluation: EvaluationSelection,
    dependencies: ModelPipelineDependencies | None = None,
    session_factory: Callable[..., Any] | None = None,
) -> MobileNetV2P43BEvaluationRecord:
    """Evaluate one verified artifact and preserve every sample outcome."""

    path = Path(model_path)
    _verify_artifact(path, artifact)
    _verify_selection(evaluation)
    modules = dependencies or load_model_pipeline_dependencies()
    factory = session_factory or modules.onnxruntime.InferenceSession
    try:
        session = factory(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise EvaluationError(f"Cannot open evaluation model with ONNX Runtime: {exc}") from exc
    input_name, output_name = _require_runtime_contract(session)
    spec = _image_input_spec(pipeline.preprocessing)
    outcomes: list[MobileNetV2P43BSampleEvaluation] = []
    for sample in evaluation.samples:
        label = sample.declaration.label
        if label is None or label > 999:
            raise EvaluationError(
                f"Evaluation sample {sample.declaration.id} has an invalid label"
            )
        try:
            tensor, _ = preprocess_image(
                sample.resolved_path,
                spec,
                numpy=modules.numpy,
                pillow_image=modules.pillow_image,
            )
        except InferenceError:
            outcomes.append(_failed_sample(sample, label, "preprocessing_failed"))
            continue
        try:
            raw_outputs = session.run([output_name], {input_name: tensor})
        except Exception:
            outcomes.append(_failed_sample(sample, label, "inference_failed"))
            continue
        outcome = _classify_output(sample, label, raw_outputs, modules.numpy)
        outcomes.append(outcome)

    records = tuple(outcomes)
    successful = sum(item.succeeded for item in records)
    failures = len(records) - successful
    non_finite = sum(item.failure == "non_finite_output" for item in records)
    top1_correct = sum(item.top1_correct for item in records)
    top5_correct = sum(item.top5_correct for item in records)
    count = len(records)
    summary = MobileNetV2P43BEvaluationSummary(
        sample_count=count,
        successful_inferences=successful,
        inference_failures=failures,
        non_finite_outputs=non_finite,
        top1_correct=top1_correct,
        top5_correct=top5_correct,
        top1_accuracy_ratio=top1_correct / count,
        top5_accuracy_ratio=top5_correct / count,
        all_outputs_finite=failures == 0 and non_finite == 0,
    )
    return MobileNetV2P43BEvaluationRecord(
        model=artifact,
        pipeline_config_sha256=sha256_canonical_json(pipeline),
        evaluation_manifest_sha256=evaluation.record.manifest_sha256,
        preprocessing_contract=pipeline.preprocessing.contract,
        execution_provider="CPUExecutionProvider",
        onnxruntime_version=str(modules.onnxruntime.__version__),
        selection=evaluation.record,
        samples=records,
        summary=summary,
    )


def _verify_artifact(path: Path, artifact: MobileNetV2P43BEvaluationArtifact) -> None:
    if path.name != artifact.filename:
        raise EvaluationError(
            f"Evaluation model filename mismatch: expected {artifact.filename}, got {path.name}"
        )
    try:
        size = path.stat().st_size
        digest = sha256_file(path)
    except (OSError, PipelineIOError) as exc:
        raise EvaluationError(f"Cannot verify evaluation model: {exc}") from exc
    if not path.is_file():
        raise EvaluationError("Evaluation model must be a regular file")
    if size != artifact.size_bytes or digest != artifact.sha256:
        raise EvaluationError("Evaluation model identity mismatch")


def _verify_selection(evaluation: EvaluationSelection) -> None:
    if evaluation.dataset.manifest.dataset.purpose != "evaluation":
        raise EvaluationError("Evaluation requires evaluation-purpose data")
    ids = tuple(sample.declaration.id for sample in evaluation.samples)
    if evaluation.record.sample_ids != ids:
        raise EvaluationError("Evaluation samples do not match the selection record")
    if evaluation.record.manifest_sha256 != evaluation.dataset.record.manifest_sha256:
        raise EvaluationError("Evaluation selection does not match its dataset record")


def _require_runtime_contract(session: Any) -> tuple[str, str]:
    try:
        providers = session.get_providers()
        inputs = session.get_inputs()
        outputs = session.get_outputs()
    except Exception as exc:
        raise EvaluationError(f"Cannot inspect ONNX Runtime session: {exc}") from exc
    if providers != ["CPUExecutionProvider"]:
        raise EvaluationError(
            "Evaluation session must use only CPUExecutionProvider, got "
            f"{providers!r}"
        )
    if len(inputs) != 1:
        raise EvaluationError("Evaluation model input contract mismatch")
    input_shape = tuple(inputs[0].shape)
    if (
        inputs[0].type != "tensor(float)"
        or len(input_shape) != 4
        or input_shape[1:] != (3, 224, 224)
    ):
        raise EvaluationError("Evaluation model input contract mismatch")
    if len(outputs) != 1:
        raise EvaluationError("Evaluation model output contract mismatch")
    output_shape = tuple(outputs[0].shape)
    if (
        outputs[0].type != "tensor(float)"
        or len(output_shape) != 2
        or output_shape[1:] != (1000,)
    ):
        raise EvaluationError("Evaluation model output contract mismatch")
    if not _batch_dimensions_compatible(input_shape[0], output_shape[0]):
        raise EvaluationError("Evaluation model batch contract mismatch")
    return inputs[0].name, outputs[0].name


def _failed_sample(
    sample: ResolvedDatasetSample,
    label: int,
    failure: FailureCode,
) -> MobileNetV2P43BSampleEvaluation:
    return MobileNetV2P43BSampleEvaluation(
        sample_id=sample.declaration.id,
        label=label,
        succeeded=False,
        finite_output=False,
        top1_class=None,
        top5_classes=None,
        top1_correct=False,
        top5_correct=False,
        failure=failure,
    )


def _classify_output(
    sample: ResolvedDatasetSample,
    label: int,
    raw_outputs: Any,
    numpy: Any,
) -> MobileNetV2P43BSampleEvaluation:
    try:
        if len(raw_outputs) != 1:
            return _failed_sample(sample, label, "invalid_output_shape")
        scores = numpy.asarray(raw_outputs[0])
        if tuple(scores.shape) != (1, 1000):
            return _failed_sample(sample, label, "invalid_output_shape")
        if not bool(numpy.isfinite(scores).all()):
            return _failed_sample(sample, label, "non_finite_output")
        ranking = numpy.argsort(-scores[0], kind="stable")[:5]
        top5 = tuple(int(value) for value in ranking)
    except Exception:
        return _failed_sample(sample, label, "invalid_output_shape")
    return MobileNetV2P43BSampleEvaluation(
        sample_id=sample.declaration.id,
        label=label,
        succeeded=True,
        finite_output=True,
        top1_class=top5[0],
        top5_classes=top5,
        top1_correct=top5[0] == label,
        top5_correct=label in top5,
        failure=None,
    )
