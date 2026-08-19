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
from rvai.model_pipeline.schema import MobileNetV2P43BAcceptanceV3Config

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


def test_v3_gates_pass_exact_agreement_and_regression_boundaries() -> None:
    fp32_samples = []
    int8_samples = []

    def top5(prediction: int, label: int, index: int) -> tuple[int, ...]:
        tail = (2000 + index, 3000 + index, 4000 + index, 5000 + index)
        return (
            (prediction, *tail)
            if prediction == label
            else (prediction, label, *tail[:3])
        )

    for index in range(100):
        label = index
        if index < 3:
            fp32_prediction, int8_prediction = label, 1000 + index
        elif index < 6:
            fp32_prediction, int8_prediction = 1000 + index, label
        elif index < 10:
            fp32_prediction, int8_prediction = 1000 + index, 1100 + index
        else:
            fp32_prediction = int8_prediction = label
        fp32_samples.append(
            sample(
                f"sample-{index}",
                label,
                fp32_prediction,
                top5(fp32_prediction, label, index),
            )
        )
        int8_samples.append(
            sample(
                f"sample-{index}",
                label,
                int8_prediction,
                top5(int8_prediction, label, index),
            )
        )
    acceptance_v3 = MobileNetV2P43BAcceptanceV3Config(
        revision="v3",
        max_top1_drop_percentage_points=1.0,
        max_top5_drop_percentage_points=1.0,
        min_model_size_reduction_ratio=0.50,
        min_top1_agreement_ratio=0.90,
        max_correct_to_wrong_regression_ratio=0.03,
        min_mean_top5_overlap_ratio=0.85,
        require_zero_inference_failures=True,
        require_finite_outputs=True,
    )

    record = compare_evaluations(
        evaluation("fp32", 1000, fp32_samples),
        evaluation("int8", 400, int8_samples),
        acceptance_v3,
    )

    assert record.top1_agreement_ratio == 0.90
    assert record.decision.correct_to_wrong_regression_count == 3
    assert record.decision.correct_to_wrong_regression_ratio == 0.03
    assert record.decision.correct_to_wrong_regression_passed is True
    assert record.decision.mean_top5_overlap_passed is True
    assert record.decision.overall_passed is True


def test_comparison_rejects_different_manifests() -> None:
    samples = (sample("sample-0", 0, 0, (0, 1, 2, 3, 4)),)

    with pytest.raises(ComparisonError, match="different dataset manifests"):
        compare_evaluations(
            evaluation("fp32", 1000, samples),
            evaluation("int8", 400, samples, "e" * 64),
            acceptance(),
        )
