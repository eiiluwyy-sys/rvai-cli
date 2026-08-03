"""Non-public synthetic proxy-pilot orchestration for MobileNetV2 P4.3B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import PositiveInt, field_validator

from rvai.model_pipeline.calibration import (
    ModelPipelineDependencies,
    load_model_pipeline_dependencies,
    select_calibration_samples,
)
from rvai.model_pipeline.compare import (
    MobileNetV2P43BComparisonRecord,
    compare_evaluations,
)
from rvai.model_pipeline.config import load_mobilenet_v2_configuration
from rvai.model_pipeline.dataset import (
    MobileNetV2P43BOverlapReport,
    require_no_dataset_overlap,
    validate_dataset,
)
from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.evaluate import (
    MobileNetV2P43BEvaluationRecord,
    evaluate_model,
    fp32_evaluation_artifact,
    int8_evaluation_artifact,
    select_evaluation_samples,
)
from rvai.model_pipeline.inspect import (
    MobileNetV2P43BSourceInspectionRecord,
    inspect_source_model,
)
from rvai.model_pipeline.io import sha256_canonical_json, write_canonical_json
from rvai.model_pipeline.quantize import (
    MobileNetV2P43BQuantizationRecord,
    quantize_static_qdq,
)
from rvai.model_pipeline.schema import (
    Description,
    MobileNetV2P43BConfiguration,
    Sha256Digest,
    StrictModel,
)
from rvai.model_pipeline.synthetic import (
    MobileNetV2P43BPseudoLabelRecord,
    MobileNetV2P43BSyntheticGenerationRecord,
    generate_synthetic_dataset,
    pseudo_label_evaluation_dataset,
)


class ProxyPilotError(ModelPipelineError):
    """Raised when the synthetic proxy pilot cannot complete safely."""


class MobileNetV2P43BProxyPilotReport(StrictModel):
    """Deterministic provisional report that cannot claim production verification."""

    schema_version: Literal["1.0"] = "1.0"
    report_type: Literal["synthetic-consistency-pilot"]
    status: Literal["provisional"]
    production_verified: Literal[False]
    label_source: Literal["fp32-top1-pseudo-label"]
    calibration_sample_count: PositiveInt
    evaluation_sample_count: PositiveInt
    pipeline_config_sha256: Sha256Digest
    source_inspection_sha256: Sha256Digest
    generation_record_sha256: Sha256Digest
    pseudo_label_record_sha256: Sha256Digest
    overlap_report_sha256: Sha256Digest
    quantization_record_sha256: Sha256Digest
    fp32_evaluation_sha256: Sha256Digest
    int8_evaluation_sha256: Sha256Digest
    comparison_sha256: Sha256Digest
    proxy_acceptance_passed: bool
    limitations: tuple[Description, ...]

    @field_validator("limitations", mode="before")
    @classmethod
    def limitation_list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


@dataclass(frozen=True)
class ProxyPilotResult:
    """Runtime result objects for a completed proxy pilot."""

    report: MobileNetV2P43BProxyPilotReport
    source_inspection: MobileNetV2P43BSourceInspectionRecord
    generation: MobileNetV2P43BSyntheticGenerationRecord
    pseudo_labels: MobileNetV2P43BPseudoLabelRecord
    overlap: MobileNetV2P43BOverlapReport
    quantization: MobileNetV2P43BQuantizationRecord
    fp32_evaluation: MobileNetV2P43BEvaluationRecord
    int8_evaluation: MobileNetV2P43BEvaluationRecord
    comparison: MobileNetV2P43BComparisonRecord


def run_synthetic_proxy_pilot(
    source_model_path: Path | str,
    output_directory: Path | str,
    *,
    configuration_directory: Path | str | None = None,
    configuration: MobileNetV2P43BConfiguration | None = None,
    calibration_sample_count: int | None = None,
    evaluation_sample_count: int | None = None,
    seed: int = 4302,
    dependencies: ModelPipelineDependencies | None = None,
) -> ProxyPilotResult:
    """Run B1 through B3 with synthetic inputs and an explicit proxy verdict."""

    if (configuration_directory is None) == (configuration is None):
        raise ProxyPilotError(
            "Supply exactly one of configuration_directory or configuration"
        )
    if configuration is not None:
        loaded = configuration
    else:
        assert configuration_directory is not None
        loaded = load_mobilenet_v2_configuration(configuration_directory)
    pipeline = loaded.pipeline
    calibration_count = (
        pipeline.calibration.pilot_samples
        if calibration_sample_count is None
        else calibration_sample_count
    )
    evaluation_count = (
        pipeline.evaluation.pilot_samples
        if evaluation_sample_count is None
        else evaluation_sample_count
    )
    modules = dependencies or load_model_pipeline_dependencies()

    output = Path(output_directory)
    try:
        output.mkdir()
        records_directory = output / "records"
        records_directory.mkdir()
        models_directory = output / "models"
        models_directory.mkdir()
    except FileExistsError as exc:
        raise ProxyPilotError(
            f"Refusing to overwrite proxy-pilot output: {output}"
        ) from exc
    except OSError as exc:
        raise ProxyPilotError(f"Cannot create proxy-pilot output: {exc}") from exc

    inspection = inspect_source_model(
        source_model_path,
        loaded.source.model,
        onnx_module=modules.onnx,
    )
    _write_record(records_directory, "source-inspection.json", inspection)

    synthetic = generate_synthetic_dataset(
        output / "dataset",
        calibration_sample_count=calibration_count,
        evaluation_sample_count=evaluation_count,
        seed=seed,
    )
    calibration_dataset = validate_dataset(
        synthetic.calibration_manifest,
        synthetic.root,
    )
    unlabeled_evaluation = validate_dataset(
        synthetic.unlabeled_evaluation_manifest,
        synthetic.root,
    )
    _write_record(
        records_directory,
        "calibration-validation.json",
        calibration_dataset.record,
    )

    fp32_artifact = fp32_evaluation_artifact(inspection)
    pseudo_labeled = pseudo_label_evaluation_dataset(
        source_model_path,
        artifact=fp32_artifact,
        pipeline=pipeline,
        unlabeled_evaluation=unlabeled_evaluation,
        destination_manifest=synthetic.root / "evaluation-manifest.json",
        dependencies=modules,
    )
    evaluation_dataset = validate_dataset(pseudo_labeled.manifest, synthetic.root)
    _write_record(
        records_directory,
        "evaluation-validation.json",
        evaluation_dataset.record,
    )
    overlap = require_no_dataset_overlap(calibration_dataset, evaluation_dataset)
    _write_record(records_directory, "overlap.json", overlap)

    calibration = select_calibration_samples(
        calibration_dataset,
        calibration_count,
    )
    int8_path = models_directory / "mobilenetv2-12-int8.onnx"
    quantization = quantize_static_qdq(
        source_model_path,
        int8_path,
        source_inspection=inspection,
        pipeline=pipeline,
        calibration=calibration,
        dependencies=modules,
    )
    _write_record(records_directory, "quantization.json", quantization)

    evaluation = select_evaluation_samples(evaluation_dataset, evaluation_count)
    fp32_evaluation = evaluate_model(
        source_model_path,
        artifact=fp32_artifact,
        pipeline=pipeline,
        evaluation=evaluation,
        dependencies=modules,
    )
    _write_record(records_directory, "fp32-evaluation.json", fp32_evaluation)
    int8_evaluation = evaluate_model(
        int8_path,
        artifact=int8_evaluation_artifact(quantization),
        pipeline=pipeline,
        evaluation=evaluation,
        dependencies=modules,
    )
    _write_record(records_directory, "int8-evaluation.json", int8_evaluation)
    comparison = compare_evaluations(
        fp32_evaluation,
        int8_evaluation,
        pipeline.acceptance,
    )
    _write_record(records_directory, "comparison.json", comparison)

    report = MobileNetV2P43BProxyPilotReport(
        report_type="synthetic-consistency-pilot",
        status="provisional",
        production_verified=False,
        label_source="fp32-top1-pseudo-label",
        calibration_sample_count=calibration_count,
        evaluation_sample_count=evaluation_count,
        pipeline_config_sha256=sha256_canonical_json(pipeline),
        source_inspection_sha256=sha256_canonical_json(inspection),
        generation_record_sha256=sha256_canonical_json(synthetic.record),
        pseudo_label_record_sha256=sha256_canonical_json(pseudo_labeled.record),
        overlap_report_sha256=sha256_canonical_json(overlap),
        quantization_record_sha256=sha256_canonical_json(quantization),
        fp32_evaluation_sha256=sha256_canonical_json(fp32_evaluation),
        int8_evaluation_sha256=sha256_canonical_json(int8_evaluation),
        comparison_sha256=sha256_canonical_json(comparison),
        proxy_acceptance_passed=comparison.decision.overall_passed,
        limitations=(
            "Synthetic inputs are not representative of ImageNet deployment data.",
            "FP32 Top-1 pseudo-labels are not independent ground-truth labels.",
            "The report measures pipeline integrity and quantization consistency only.",
            "The report is not a physical RISC-V performance benchmark.",
        ),
    )
    _write_record(records_directory, "proxy-pilot-report.json", report)
    return ProxyPilotResult(
        report=report,
        source_inspection=inspection,
        generation=synthetic.record,
        pseudo_labels=pseudo_labeled.record,
        overlap=overlap,
        quantization=quantization,
        fp32_evaluation=fp32_evaluation,
        int8_evaluation=int8_evaluation,
        comparison=comparison,
    )


def _write_record(directory: Path, filename: str, record: StrictModel) -> None:
    write_canonical_json(directory / filename, record)
