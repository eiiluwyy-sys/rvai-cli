import pytest

from rvai.model_pipeline.compare import ComparisonError, compare_evaluations
from rvai.model_pipeline.config import load_pipeline_config
from rvai.model_pipeline.evaluate import (
    MobileNetV2P43BEvaluationArtifact,
    MobileNetV2P43BEvaluationRecord,
    MobileNetV2P43BEvaluationSelectionRecord,
    MobileNetV2P43BEvaluationSummary,
    MobileNetV2P43BSampleEvaluation,
)


DIGEST = "a" * 64


def sample(sample_id: str, label: int, prediction: int, top5: tuple[int, ...]):
    return MobileNetV2P43BSampleEvaluation(
        sample_id=sample_id,
        label=label,
        succeeded=True,
        finite_output=True,
        top1_class=prediction,
        top5_classes=top5,
        top1_correct=prediction == label,
        top5_correct=label in top5,
        failure=None,
    )


def evaluation(role: str, size: int, samples, manifest_sha: str = DIGEST):
    samples = tuple(samples)
    count = len(samples)
    top1 = sum(item.top1_correct for item in samples)
    top5 = sum(item.top5_correct for item in samples)
    selection = MobileNetV2P43BEvaluationSelectionRecord(
        manifest_sha256=manifest_sha,
        sample_order="manifest",
        requested_sample_count=count,
        actual_sample_count=count,
        used_complete_available_set=False,
        sample_ids=tuple(item.sample_id for item in samples),
    )
    return MobileNetV2P43BEvaluationRecord(
        model=MobileNetV2P43BEvaluationArtifact(
            role=role,
            filename=f"model-{role}.onnx",
            size_bytes=size,
            sha256=("b" if role == "fp32" else "c") * 64,
        ),
        pipeline_config_sha256="d" * 64,
        evaluation_manifest_sha256=manifest_sha,
        preprocessing_contract="mobilenet-v2-imagenet-v1",
        execution_provider="CPUExecutionProvider",
        onnxruntime_version="1.23.2",
        selection=selection,
        samples=samples,
        summary=MobileNetV2P43BEvaluationSummary(
            sample_count=count,
            successful_inferences=count,
            inference_failures=0,
            non_finite_outputs=0,
            top1_correct=top1,
            top5_correct=top5,
            top1_accuracy_ratio=top1 / count,
            top5_accuracy_ratio=top5 / count,
            all_outputs_finite=True,
        ),
    )


def acceptance():
    from pathlib import Path

    config = Path(__file__).parents[1] / "model-pipeline" / "mobilenet-v2" / "pipeline.yaml"
    return load_pipeline_config(config).acceptance


def test_comparison_calculates_metrics_and_each_gate() -> None:
    fp32 = evaluation(
        "fp32",
        1000,
        (
            sample("sample-0", 0, 0, (0, 1, 2, 3, 4)),
            sample("sample-1", 1, 1, (1, 0, 2, 3, 4)),
        ),
    )
    int8 = evaluation(
        "int8",
        400,
        (
            sample("sample-0", 0, 0, (0, 1, 2, 3, 4)),
            sample("sample-1", 1, 2, (2, 1, 0, 3, 4)),
        ),
    )

    record = compare_evaluations(fp32, int8, acceptance())

    assert record.top1_drop_percentage_points == 50.0
    assert record.top5_drop_percentage_points == 0.0
    assert record.model_size_reduction_ratio == 0.6
    assert record.top1_agreement_ratio == 0.5
    assert record.mean_top5_overlap_ratio == 1.0
    assert record.decision.top1_accuracy_passed is False
    assert record.decision.top5_accuracy_passed is True
    assert record.decision.model_size_passed is True
    assert record.decision.top1_agreement_passed is False
    assert record.decision.overall_passed is False


def test_comparison_passes_identical_accurate_predictions() -> None:
    samples = (
        sample("sample-0", 0, 0, (0, 1, 2, 3, 4)),
        sample("sample-1", 1, 1, (1, 0, 2, 3, 4)),
    )

    record = compare_evaluations(
        evaluation("fp32", 1000, samples),
        evaluation("int8", 400, samples),
        acceptance(),
    )

    assert record.decision.overall_passed is True


def test_one_percentage_point_accuracy_drop_passes_exact_boundary() -> None:
    fp32_samples = tuple(
        sample(f"sample-{index}", index, index, (index, 100, 101, 102, 103))
        for index in range(100)
    )
    int8_samples = fp32_samples[:-1] + (
        sample("sample-99", 99, 0, (0, 1, 2, 3, 4)),
    )

    record = compare_evaluations(
        evaluation("fp32", 1000, fp32_samples),
        evaluation("int8", 400, int8_samples),
        acceptance(),
    )

    assert record.top1_drop_percentage_points == 1.0
    assert record.decision.top1_accuracy_passed is True


def test_comparison_rejects_different_manifests() -> None:
    samples = (sample("sample-0", 0, 0, (0, 1, 2, 3, 4)),)

    with pytest.raises(ComparisonError, match="different dataset manifests"):
        compare_evaluations(
            evaluation("fp32", 1000, samples),
            evaluation("int8", 400, samples, "e" * 64),
            acceptance(),
        )
