"""Accuracy, size, and output-consistency comparison for P4.3B."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.evaluate import MobileNetV2P43BEvaluationRecord
from rvai.model_pipeline.io import sha256_canonical_json
from rvai.model_pipeline.schema import (
    MobileNetV2P43BAcceptanceConfig,
    Sha256Digest,
    StrictModel,
)


class ComparisonError(ModelPipelineError):
    """Raised when evaluation records cannot be compared safely."""


class MobileNetV2P43BAcceptanceDecision(StrictModel):
    """Independent decisions for every frozen engineering gate."""

    top1_accuracy_passed: bool
    top5_accuracy_passed: bool
    model_size_passed: bool
    top1_agreement_passed: bool
    zero_inference_failures_passed: bool
    finite_outputs_passed: bool
    overall_passed: bool

    @model_validator(mode="after")
    def overall_matches_gates(self) -> "MobileNetV2P43BAcceptanceDecision":
        gates = (
            self.top1_accuracy_passed,
            self.top5_accuracy_passed,
            self.model_size_passed,
            self.top1_agreement_passed,
            self.zero_inference_failures_passed,
            self.finite_outputs_passed,
        )
        if self.overall_passed != all(gates):
            raise ValueError("overall_passed must equal all individual gates")
        return self


class MobileNetV2P43BComparisonRecord(StrictModel):
    """Deterministic FP32-to-INT8 comparison and acceptance evidence."""

    schema_version: Literal["1.0"] = "1.0"
    fp32_evaluation_sha256: Sha256Digest
    int8_evaluation_sha256: Sha256Digest
    evaluation_manifest_sha256: Sha256Digest
    sample_count: PositiveInt
    fp32_top1_accuracy_ratio: float = Field(ge=0.0, le=1.0)
    int8_top1_accuracy_ratio: float = Field(ge=0.0, le=1.0)
    top1_drop_percentage_points: float
    fp32_top5_accuracy_ratio: float = Field(ge=0.0, le=1.0)
    int8_top5_accuracy_ratio: float = Field(ge=0.0, le=1.0)
    top5_drop_percentage_points: float
    fp32_model_size_bytes: PositiveInt
    int8_model_size_bytes: PositiveInt
    model_size_reduction_ratio: float
    top1_agreement_count: NonNegativeInt
    top1_agreement_ratio: float = Field(ge=0.0, le=1.0)
    mean_top5_overlap_ratio: float = Field(ge=0.0, le=1.0)
    total_inference_failures: NonNegativeInt
    all_outputs_finite: bool
    acceptance: MobileNetV2P43BAcceptanceConfig
    decision: MobileNetV2P43BAcceptanceDecision


def compare_evaluations(
    fp32: MobileNetV2P43BEvaluationRecord,
    int8: MobileNetV2P43BEvaluationRecord,
    acceptance: MobileNetV2P43BAcceptanceConfig,
) -> MobileNetV2P43BComparisonRecord:
    """Compare paired manifest-order results and evaluate every frozen gate."""

    _require_comparable(fp32, int8)
    sample_count = len(fp32.samples)
    top1_agreement_count = 0
    top5_overlap_count = 0
    for fp32_sample, int8_sample in zip(fp32.samples, int8.samples, strict=True):
        if (
            fp32_sample.succeeded
            and int8_sample.succeeded
            and fp32_sample.top1_class == int8_sample.top1_class
        ):
            top1_agreement_count += 1
        if (
            fp32_sample.top5_classes is not None
            and int8_sample.top5_classes is not None
        ):
            overlap = set(fp32_sample.top5_classes) & set(int8_sample.top5_classes)
            top5_overlap_count += len(overlap)

    fp32_top1 = fp32.summary.top1_accuracy_ratio
    int8_top1 = int8.summary.top1_accuracy_ratio
    fp32_top5 = fp32.summary.top5_accuracy_ratio
    int8_top5 = int8.summary.top5_accuracy_ratio
    top1_drop = (
        (fp32.summary.top1_correct - int8.summary.top1_correct)
        * 100.0
        / sample_count
    )
    top5_drop = (
        (fp32.summary.top5_correct - int8.summary.top5_correct)
        * 100.0
        / sample_count
    )
    size_reduction = (
        (fp32.model.size_bytes - int8.model.size_bytes) / fp32.model.size_bytes
    )
    agreement_ratio = top1_agreement_count / sample_count
    mean_top5_overlap = top5_overlap_count / (5 * sample_count)
    total_failures = (
        fp32.summary.inference_failures + int8.summary.inference_failures
    )
    all_finite = fp32.summary.all_outputs_finite and int8.summary.all_outputs_finite
    gates = (
        top1_drop <= acceptance.max_top1_drop_percentage_points,
        top5_drop <= acceptance.max_top5_drop_percentage_points,
        size_reduction >= acceptance.min_model_size_reduction_ratio,
        agreement_ratio >= acceptance.min_top1_agreement_ratio,
        total_failures == 0 if acceptance.require_zero_inference_failures else True,
        all_finite if acceptance.require_finite_outputs else True,
    )
    decision = MobileNetV2P43BAcceptanceDecision(
        top1_accuracy_passed=gates[0],
        top5_accuracy_passed=gates[1],
        model_size_passed=gates[2],
        top1_agreement_passed=gates[3],
        zero_inference_failures_passed=gates[4],
        finite_outputs_passed=gates[5],
        overall_passed=all(gates),
    )
    return MobileNetV2P43BComparisonRecord(
        fp32_evaluation_sha256=sha256_canonical_json(fp32),
        int8_evaluation_sha256=sha256_canonical_json(int8),
        evaluation_manifest_sha256=fp32.evaluation_manifest_sha256,
        sample_count=sample_count,
        fp32_top1_accuracy_ratio=fp32_top1,
        int8_top1_accuracy_ratio=int8_top1,
        top1_drop_percentage_points=top1_drop,
        fp32_top5_accuracy_ratio=fp32_top5,
        int8_top5_accuracy_ratio=int8_top5,
        top5_drop_percentage_points=top5_drop,
        fp32_model_size_bytes=fp32.model.size_bytes,
        int8_model_size_bytes=int8.model.size_bytes,
        model_size_reduction_ratio=size_reduction,
        top1_agreement_count=top1_agreement_count,
        top1_agreement_ratio=agreement_ratio,
        mean_top5_overlap_ratio=mean_top5_overlap,
        total_inference_failures=total_failures,
        all_outputs_finite=all_finite,
        acceptance=acceptance,
        decision=decision,
    )


def _require_comparable(
    fp32: MobileNetV2P43BEvaluationRecord,
    int8: MobileNetV2P43BEvaluationRecord,
) -> None:
    if fp32.model.role != "fp32" or int8.model.role != "int8":
        raise ComparisonError("Comparison requires FP32 then INT8 evaluation records")
    if fp32.pipeline_config_sha256 != int8.pipeline_config_sha256:
        raise ComparisonError("Evaluation records use different pipeline configurations")
    if fp32.evaluation_manifest_sha256 != int8.evaluation_manifest_sha256:
        raise ComparisonError("Evaluation records use different dataset manifests")
    if fp32.selection != int8.selection:
        raise ComparisonError("Evaluation records use different sample selections")
    if len(fp32.samples) == 0 or len(fp32.samples) != len(int8.samples):
        raise ComparisonError("Evaluation records must contain the same non-empty samples")
    fp32_pairs = tuple((item.sample_id, item.label) for item in fp32.samples)
    int8_pairs = tuple((item.sample_id, item.label) for item in int8.samples)
    if fp32_pairs != int8_pairs:
        raise ComparisonError("Evaluation sample identities or labels do not match")
