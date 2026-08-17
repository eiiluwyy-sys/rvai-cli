"""Formal independently labelled MobileNetV2 P4.3B production orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import PositiveInt, field_validator, model_validator

from rvai.model_pipeline.calibration import (
    ModelPipelineDependencies,
    load_model_pipeline_dependencies,
    select_calibration_samples,
)
from rvai.model_pipeline.compare import (
    MobileNetV2P43BComparisonRecord,
    compare_evaluations,
)
from rvai.model_pipeline.config import (
    load_dataset_manifest,
    load_mobilenet_v2_configuration,
)
from rvai.model_pipeline.dataset import (
    MobileNetV2P43BDatasetValidationRecord,
    MobileNetV2P43BOverlapReport,
    require_no_dataset_overlap,
    validate_dataset,
)
from rvai.model_pipeline.environment import (
    MobileNetV2P43BExecutionEnvironment,
    MobileNetV2P43BSoftwareEnvironment,
    MobileNetV2P43BSourceRevision,
    collect_execution_environment,
    collect_software_environment,
    collect_source_revision,
)
from rvai.model_pipeline.errors import ModelPipelineError, PipelineIOError
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
from rvai.model_pipeline.io import (
    canonical_json_bytes,
    load_json,
    sha256_canonical_json,
    sha256_file,
    write_canonical_json,
)
from rvai.model_pipeline.quantize import (
    MobileNetV2P43BQuantizationRecord,
    quantize_static_qdq,
)
from rvai.model_pipeline.schema import (
    CanonicalRelativePath,
    Description,
    MobileNetV2P43BConfiguration,
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BPipelineConfig,
    MobileNetV2P43BSourceModelConfig,
    Sha256Digest,
    StrictModel,
)


class ProductionRunError(ModelPipelineError):
    """Raised when a formal production run cannot complete safely."""


ProductionStage = Literal[
    "configuration",
    "source-revision",
    "source-inspection",
    "dataset-validation",
    "dataset-overlap",
    "quantization",
    "evaluation",
    "comparison",
    "input-revalidation",
    "reproducibility",
]


class MobileNetV2P43BProductionFailureRecord(StrictModel):
    """Non-deterministic run envelope retained after an operational failure."""

    schema_version: Literal["1.0"] = "1.0"
    report_type: Literal["formal-production-validation"]
    status: Literal["failed"]
    production_verified: Literal[False]
    label_source: Literal["independent-ground-truth"]
    failed_stage: ProductionStage
    error_type: Description
    message: Description
    completed_records: tuple[CanonicalRelativePath, ...]

    @field_validator("completed_records", mode="before")
    @classmethod
    def record_list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class MobileNetV2P43BProductionReport(StrictModel):
    """Deterministic acceptance report for independently labelled data."""

    schema_version: Literal["1.0"] = "1.0"
    report_type: Literal["formal-production-validation"]
    status: Literal["accepted", "rejected"]
    production_verified: bool
    label_source: Literal["independent-ground-truth"]
    calibration_requested_sample_count: PositiveInt
    calibration_actual_sample_count: PositiveInt
    evaluation_requested_sample_count: PositiveInt
    evaluation_actual_sample_count: PositiveInt
    evaluation_used_complete_available_set: bool
    pipeline_config_sha256: Sha256Digest
    source_inspection_sha256: Sha256Digest
    calibration_manifest_sha256: Sha256Digest
    evaluation_manifest_sha256: Sha256Digest
    overlap_report_sha256: Sha256Digest
    quantization_record_sha256: Sha256Digest
    fp32_evaluation_sha256: Sha256Digest
    int8_evaluation_sha256: Sha256Digest
    comparison_sha256: Sha256Digest
    acceptance_passed: bool

    @model_validator(mode="after")
    def status_matches_acceptance(self) -> "MobileNetV2P43BProductionReport":
        accepted = self.status == "accepted"
        if accepted != self.acceptance_passed:
            raise ValueError("status must match acceptance_passed")
        if self.production_verified != self.acceptance_passed:
            raise ValueError("production_verified must match acceptance_passed")
        if (
            self.evaluation_used_complete_available_set
            != (
                self.evaluation_actual_sample_count
                < self.evaluation_requested_sample_count
            )
        ):
            raise ValueError("evaluation fallback flag does not match sample counts")
        return self


class MobileNetV2P43BProductionInputDigests(StrictModel):
    """Canonical identities of formal production inputs."""

    pipeline_config_sha256: Sha256Digest
    source_model_config_sha256: Sha256Digest
    source_fp32_model_sha256: Sha256Digest
    calibration_manifest_sha256: Sha256Digest
    evaluation_manifest_sha256: Sha256Digest


class MobileNetV2P43BProductionOutputDigests(StrictModel):
    """Canonical identities of formal production outputs."""

    source_inspection_sha256: Sha256Digest
    calibration_validation_sha256: Sha256Digest
    evaluation_validation_sha256: Sha256Digest
    overlap_report_sha256: Sha256Digest
    quantization_record_sha256: Sha256Digest
    int8_model_sha256: Sha256Digest
    fp32_evaluation_sha256: Sha256Digest
    int8_evaluation_sha256: Sha256Digest
    comparison_sha256: Sha256Digest
    production_report_sha256: Sha256Digest


class MobileNetV2P43BProductionReproducibilityRecord(StrictModel):
    """Environment and complete digest graph for one formal run."""

    schema_version: Literal["1.0"] = "1.0"
    pipeline: Literal["mobilenet-v2-int8"]
    report_type: Literal["formal-production-validation"]
    status: Literal["accepted", "rejected"]
    production_verified: bool
    label_source: Literal["independent-ground-truth"]
    software: MobileNetV2P43BSoftwareEnvironment
    execution: MobileNetV2P43BExecutionEnvironment
    source_revision: MobileNetV2P43BSourceRevision
    inputs: MobileNetV2P43BProductionInputDigests
    outputs: MobileNetV2P43BProductionOutputDigests

    @model_validator(mode="after")
    def status_matches_verification(
        self,
    ) -> "MobileNetV2P43BProductionReproducibilityRecord":
        if (self.status == "accepted") != self.production_verified:
            raise ValueError("status must match production_verified")
        if not self.source_revision.working_tree_clean:
            raise ValueError("formal production requires a clean source revision")
        return self


@dataclass(frozen=True)
class ProductionRunResult:
    """Runtime objects from one completed accepted or rejected formal run."""

    report: MobileNetV2P43BProductionReport
    source_inspection: MobileNetV2P43BSourceInspectionRecord
    overlap: MobileNetV2P43BOverlapReport
    quantization: MobileNetV2P43BQuantizationRecord
    fp32_evaluation: MobileNetV2P43BEvaluationRecord
    int8_evaluation: MobileNetV2P43BEvaluationRecord
    comparison: MobileNetV2P43BComparisonRecord
    reproducibility: MobileNetV2P43BProductionReproducibilityRecord


@dataclass(frozen=True)
class ValidatedProductionEvidence:
    """Strictly loaded portable records and verified generated artifact."""

    pipeline: MobileNetV2P43BPipelineConfig
    source: MobileNetV2P43BSourceModelConfig
    calibration_manifest: MobileNetV2P43BDatasetManifest
    evaluation_manifest: MobileNetV2P43BDatasetManifest
    source_inspection: MobileNetV2P43BSourceInspectionRecord
    calibration_validation: MobileNetV2P43BDatasetValidationRecord
    evaluation_validation: MobileNetV2P43BDatasetValidationRecord
    overlap: MobileNetV2P43BOverlapReport
    quantization: MobileNetV2P43BQuantizationRecord
    fp32_evaluation: MobileNetV2P43BEvaluationRecord
    int8_evaluation: MobileNetV2P43BEvaluationRecord
    comparison: MobileNetV2P43BComparisonRecord
    report: MobileNetV2P43BProductionReport
    reproducibility: MobileNetV2P43BProductionReproducibilityRecord
    int8_model_path: Path


PRODUCTION_RECORD_TYPES: dict[str, type[StrictModel]] = {
    "records/pipeline-config.json": MobileNetV2P43BPipelineConfig,
    "records/source-model-config.json": MobileNetV2P43BSourceModelConfig,
    "records/calibration-manifest.json": MobileNetV2P43BDatasetManifest,
    "records/evaluation-manifest.json": MobileNetV2P43BDatasetManifest,
    "records/source-inspection.json": MobileNetV2P43BSourceInspectionRecord,
    "records/calibration-validation.json": MobileNetV2P43BDatasetValidationRecord,
    "records/evaluation-validation.json": MobileNetV2P43BDatasetValidationRecord,
    "records/overlap.json": MobileNetV2P43BOverlapReport,
    "records/quantization.json": MobileNetV2P43BQuantizationRecord,
    "records/fp32-evaluation.json": MobileNetV2P43BEvaluationRecord,
    "records/int8-evaluation.json": MobileNetV2P43BEvaluationRecord,
    "records/comparison.json": MobileNetV2P43BComparisonRecord,
    "records/production-report.json": MobileNetV2P43BProductionReport,
    "records/reproducibility.json": MobileNetV2P43BProductionReproducibilityRecord,
}


def run_production_pipeline(
    source_model_path: Path | str,
    calibration_manifest_path: Path | str,
    calibration_root: Path | str,
    evaluation_manifest_path: Path | str,
    evaluation_root: Path | str,
    output_directory: Path | str,
    *,
    configuration_directory: Path | str | None = None,
    configuration: MobileNetV2P43BConfiguration | None = None,
    repository: Path | str | None = None,
    dependencies: ModelPipelineDependencies | None = None,
) -> ProductionRunResult:
    """Run the frozen production configuration against independent labels."""

    if (configuration_directory is None) == (configuration is None):
        raise ProductionRunError(
            "Supply exactly one of configuration_directory or configuration"
        )
    output = Path(output_directory)
    try:
        output.mkdir()
        records = output / "records"
        records.mkdir()
        models = output / "models"
        models.mkdir()
    except FileExistsError as exc:
        raise ProductionRunError(
            f"Refusing to overwrite production output: {output}"
        ) from exc
    except OSError as exc:
        raise ProductionRunError(f"Cannot create production output: {exc}") from exc

    stage: ProductionStage = "configuration"
    try:
        if configuration is not None:
            loaded = configuration
        else:
            assert configuration_directory is not None
            loaded = load_mobilenet_v2_configuration(configuration_directory)
        calibration_manifest = load_dataset_manifest(
            calibration_manifest_path,
            expected_purpose="calibration",
        )
        evaluation_manifest = load_dataset_manifest(
            evaluation_manifest_path,
            expected_purpose="evaluation",
        )
        _write_record(records, "pipeline-config.json", loaded.pipeline)
        _write_record(records, "source-model-config.json", loaded.source)
        _write_record(records, "calibration-manifest.json", calibration_manifest)
        _write_record(records, "evaluation-manifest.json", evaluation_manifest)

        stage = "source-revision"
        source_repository = (
            Path(__file__).resolve().parents[3]
            if repository is None
            else Path(repository)
        )
        start_revision = collect_source_revision(source_repository)
        if not start_revision.working_tree_clean:
            raise ProductionRunError(
                "Formal production requires a clean Git working tree"
            )
        modules = dependencies or load_model_pipeline_dependencies()

        stage = "source-inspection"
        inspection = inspect_source_model(
            source_model_path,
            loaded.source.model,
            onnx_module=modules.onnx,
        )
        _write_record(records, "source-inspection.json", inspection)

        stage = "dataset-validation"
        calibration_dataset = validate_dataset(calibration_manifest, calibration_root)
        evaluation_dataset = validate_dataset(evaluation_manifest, evaluation_root)
        _write_record(
            records,
            "calibration-validation.json",
            calibration_dataset.record,
        )
        _write_record(
            records,
            "evaluation-validation.json",
            evaluation_dataset.record,
        )

        stage = "dataset-overlap"
        overlap = require_no_dataset_overlap(
            calibration_dataset,
            evaluation_dataset,
        )
        _write_record(records, "overlap.json", overlap)

        stage = "quantization"
        calibration = select_calibration_samples(
            calibration_dataset,
            loaded.pipeline.calibration.production_samples,
        )
        int8_path = models / "mobilenetv2-12-int8.onnx"
        quantization = quantize_static_qdq(
            source_model_path,
            int8_path,
            source_inspection=inspection,
            pipeline=loaded.pipeline,
            calibration=calibration,
            dependencies=modules,
        )
        _write_record(records, "quantization.json", quantization)

        stage = "evaluation"
        evaluation = select_evaluation_samples(
            evaluation_dataset,
            loaded.pipeline.evaluation.production_samples,
            allow_complete_dataset_if_fewer=True,
        )
        fp32_evaluation = evaluate_model(
            source_model_path,
            artifact=fp32_evaluation_artifact(inspection),
            pipeline=loaded.pipeline,
            evaluation=evaluation,
            dependencies=modules,
        )
        _write_record(records, "fp32-evaluation.json", fp32_evaluation)
        int8_evaluation = evaluate_model(
            int8_path,
            artifact=int8_evaluation_artifact(quantization),
            pipeline=loaded.pipeline,
            evaluation=evaluation,
            dependencies=modules,
        )
        _write_record(records, "int8-evaluation.json", int8_evaluation)

        stage = "comparison"
        comparison = compare_evaluations(
            fp32_evaluation,
            int8_evaluation,
            loaded.pipeline.acceptance,
        )
        _write_record(records, "comparison.json", comparison)

        stage = "input-revalidation"
        if (
            validate_dataset(calibration_manifest, calibration_root).record
            != calibration_dataset.record
            or validate_dataset(evaluation_manifest, evaluation_root).record
            != evaluation_dataset.record
        ):
            raise ProductionRunError("Dataset content changed during production run")
        end_revision = collect_source_revision(source_repository)
        if end_revision != start_revision:
            raise ProductionRunError("Source revision changed during production run")

        accepted = comparison.decision.overall_passed
        report = MobileNetV2P43BProductionReport(
            report_type="formal-production-validation",
            status="accepted" if accepted else "rejected",
            production_verified=accepted,
            label_source="independent-ground-truth",
            calibration_requested_sample_count=(
                loaded.pipeline.calibration.production_samples
            ),
            calibration_actual_sample_count=calibration.record.sample_count,
            evaluation_requested_sample_count=evaluation.record.requested_sample_count,
            evaluation_actual_sample_count=evaluation.record.actual_sample_count,
            evaluation_used_complete_available_set=(
                evaluation.record.used_complete_available_set
            ),
            pipeline_config_sha256=sha256_canonical_json(loaded.pipeline),
            source_inspection_sha256=sha256_canonical_json(inspection),
            calibration_manifest_sha256=sha256_canonical_json(calibration_manifest),
            evaluation_manifest_sha256=sha256_canonical_json(evaluation_manifest),
            overlap_report_sha256=sha256_canonical_json(overlap),
            quantization_record_sha256=sha256_canonical_json(quantization),
            fp32_evaluation_sha256=sha256_canonical_json(fp32_evaluation),
            int8_evaluation_sha256=sha256_canonical_json(int8_evaluation),
            comparison_sha256=sha256_canonical_json(comparison),
            acceptance_passed=accepted,
        )
        _write_record(records, "production-report.json", report)

        stage = "reproducibility"
        reproducibility = MobileNetV2P43BProductionReproducibilityRecord(
            pipeline="mobilenet-v2-int8",
            report_type="formal-production-validation",
            status=report.status,
            production_verified=report.production_verified,
            label_source="independent-ground-truth",
            software=collect_software_environment(modules),
            execution=collect_execution_environment(),
            source_revision=end_revision,
            inputs=MobileNetV2P43BProductionInputDigests(
                pipeline_config_sha256=sha256_canonical_json(loaded.pipeline),
                source_model_config_sha256=sha256_canonical_json(loaded.source),
                source_fp32_model_sha256=inspection.model.sha256,
                calibration_manifest_sha256=sha256_canonical_json(
                    calibration_manifest
                ),
                evaluation_manifest_sha256=sha256_canonical_json(
                    evaluation_manifest
                ),
            ),
            outputs=MobileNetV2P43BProductionOutputDigests(
                source_inspection_sha256=sha256_canonical_json(inspection),
                calibration_validation_sha256=sha256_canonical_json(
                    calibration_dataset.record
                ),
                evaluation_validation_sha256=sha256_canonical_json(
                    evaluation_dataset.record
                ),
                overlap_report_sha256=sha256_canonical_json(overlap),
                quantization_record_sha256=sha256_canonical_json(quantization),
                int8_model_sha256=quantization.artifact.sha256,
                fp32_evaluation_sha256=sha256_canonical_json(fp32_evaluation),
                int8_evaluation_sha256=sha256_canonical_json(int8_evaluation),
                comparison_sha256=sha256_canonical_json(comparison),
                production_report_sha256=sha256_canonical_json(report),
            ),
        )
        _write_record(records, "reproducibility.json", reproducibility)
        return ProductionRunResult(
            report=report,
            source_inspection=inspection,
            overlap=overlap,
            quantization=quantization,
            fp32_evaluation=fp32_evaluation,
            int8_evaluation=int8_evaluation,
            comparison=comparison,
            reproducibility=reproducibility,
        )
    except Exception as exc:
        _retain_failure(records, stage, exc)
        if isinstance(exc, ModelPipelineError):
            raise
        raise ProductionRunError(f"Production stage {stage} failed: {exc}") from exc


def validate_production_evidence(
    run_directory: Path | str,
) -> ValidatedProductionEvidence:
    """Load and cross-check a completed formal run without modifying it."""

    requested_root = Path(run_directory)
    if requested_root.is_symlink():
        raise ProductionRunError("Production evidence root must not be a symlink")
    try:
        root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise ProductionRunError(f"Cannot resolve production evidence: {exc}") from exc
    if not root.is_dir():
        raise ProductionRunError("Production evidence root must be a directory")
    records: dict[str, StrictModel] = {}
    for relative_path, model_type in PRODUCTION_RECORD_TYPES.items():
        path = _safe_evidence_file(root, relative_path)
        try:
            value = load_json(path, model_type)
        except PipelineIOError as exc:
            raise ProductionRunError(
                f"Invalid production record {relative_path}: {exc}"
            ) from exc
        if path.read_bytes() != canonical_json_bytes(value):
            raise ProductionRunError(
                f"Production record is not canonical JSON: {relative_path}"
            )
        records[relative_path] = value
    evidence = ValidatedProductionEvidence(
        pipeline=_record(records, "pipeline-config.json", MobileNetV2P43BPipelineConfig),
        source=_record(records, "source-model-config.json", MobileNetV2P43BSourceModelConfig),
        calibration_manifest=_record(records, "calibration-manifest.json", MobileNetV2P43BDatasetManifest),
        evaluation_manifest=_record(records, "evaluation-manifest.json", MobileNetV2P43BDatasetManifest),
        source_inspection=_record(records, "source-inspection.json", MobileNetV2P43BSourceInspectionRecord),
        calibration_validation=_record(records, "calibration-validation.json", MobileNetV2P43BDatasetValidationRecord),
        evaluation_validation=_record(records, "evaluation-validation.json", MobileNetV2P43BDatasetValidationRecord),
        overlap=_record(records, "overlap.json", MobileNetV2P43BOverlapReport),
        quantization=_record(records, "quantization.json", MobileNetV2P43BQuantizationRecord),
        fp32_evaluation=_record(records, "fp32-evaluation.json", MobileNetV2P43BEvaluationRecord),
        int8_evaluation=_record(records, "int8-evaluation.json", MobileNetV2P43BEvaluationRecord),
        comparison=_record(records, "comparison.json", MobileNetV2P43BComparisonRecord),
        report=_record(records, "production-report.json", MobileNetV2P43BProductionReport),
        reproducibility=_record(records, "reproducibility.json", MobileNetV2P43BProductionReproducibilityRecord),
        int8_model_path=_safe_evidence_file(
            root,
            "models/mobilenetv2-12-int8.onnx",
        ),
    )
    _validate_production_graph(evidence)
    return evidence


def _validate_production_graph(evidence: ValidatedProductionEvidence) -> None:
    pipeline_sha = sha256_canonical_json(evidence.pipeline)
    source_sha = sha256_canonical_json(evidence.source)
    calibration_sha = sha256_canonical_json(evidence.calibration_manifest)
    evaluation_sha = sha256_canonical_json(evidence.evaluation_manifest)
    inspection_sha = sha256_canonical_json(evidence.source_inspection)
    overlap_sha = sha256_canonical_json(evidence.overlap)
    quantization_sha = sha256_canonical_json(evidence.quantization)
    fp32_sha = sha256_canonical_json(evidence.fp32_evaluation)
    int8_sha = sha256_canonical_json(evidence.int8_evaluation)
    comparison_sha = sha256_canonical_json(evidence.comparison)
    report_sha = sha256_canonical_json(evidence.report)
    calibration_validation_sha = sha256_canonical_json(
        evidence.calibration_validation
    )
    evaluation_validation_sha = sha256_canonical_json(evidence.evaluation_validation)

    if evidence.source_inspection.model != evidence.source.model:
        raise ProductionRunError("Source inspection does not match source config")
    if (
        evidence.calibration_manifest.dataset.purpose != "calibration"
        or evidence.evaluation_manifest.dataset.purpose != "evaluation"
    ):
        raise ProductionRunError("Production dataset purposes are invalid")
    _validate_dataset_link(
        evidence.calibration_manifest,
        evidence.calibration_validation,
    )
    _validate_dataset_link(
        evidence.evaluation_manifest,
        evidence.evaluation_validation,
    )
    if (
        evidence.overlap.calibration_manifest_sha256 != calibration_sha
        or evidence.overlap.evaluation_manifest_sha256 != evaluation_sha
        or evidence.overlap.overlap_count != 0
        or evidence.overlap.overlaps
    ):
        raise ProductionRunError("Dataset overlap evidence is invalid")
    quantization = evidence.quantization
    if (
        quantization.source_model_sha256 != evidence.source.model.sha256
        or quantization.source_inspection_sha256 != inspection_sha
        or quantization.pipeline_config_sha256 != pipeline_sha
        or quantization.calibration_manifest_sha256 != calibration_sha
        or quantization.quantization != evidence.pipeline.quantization
    ):
        raise ProductionRunError("Quantization record link mismatch")
    calibration_ids = tuple(
        sample.id
        for sample in evidence.calibration_validation.samples[
            : quantization.calibration_sample_count
        ]
    )
    if calibration_ids != quantization.calibration_sample_ids:
        raise ProductionRunError("Quantization calibration order mismatch")
    model = evidence.int8_model_path
    if (
        not model.is_file()
        or model.name != quantization.artifact.filename
        or model.stat().st_size != quantization.artifact.size_bytes
        or sha256_file(model) != quantization.artifact.sha256
    ):
        raise ProductionRunError("INT8 model identity mismatch")

    fp32 = evidence.fp32_evaluation
    int8 = evidence.int8_evaluation
    if (
        fp32.model.role != "fp32"
        or fp32.model.filename != evidence.source.model.filename
        or fp32.model.size_bytes != evidence.source.model.size_bytes
        or fp32.model.sha256 != evidence.source.model.sha256
        or int8.model.role != "int8"
        or int8.model.filename != quantization.artifact.filename
        or int8.model.size_bytes != quantization.artifact.size_bytes
        or int8.model.sha256 != quantization.artifact.sha256
        or fp32.pipeline_config_sha256 != pipeline_sha
        or int8.pipeline_config_sha256 != pipeline_sha
        or fp32.evaluation_manifest_sha256 != evaluation_sha
        or int8.evaluation_manifest_sha256 != evaluation_sha
        or fp32.selection != int8.selection
    ):
        raise ProductionRunError("Evaluation record link mismatch")
    expected_pairs = tuple(
        (sample.id, sample.label)
        for sample in evidence.evaluation_validation.samples[
            : fp32.selection.actual_sample_count
        ]
    )
    for evaluation in (fp32, int8):
        if tuple(
            (sample.sample_id, sample.label) for sample in evaluation.samples
        ) != expected_pairs:
            raise ProductionRunError("Evaluation manifest order mismatch")
    if compare_evaluations(fp32, int8, evidence.pipeline.acceptance) != evidence.comparison:
        raise ProductionRunError("Comparison record does not recompute exactly")

    report = evidence.report
    links = (
        (report.pipeline_config_sha256, pipeline_sha),
        (report.source_inspection_sha256, inspection_sha),
        (report.calibration_manifest_sha256, calibration_sha),
        (report.evaluation_manifest_sha256, evaluation_sha),
        (report.overlap_report_sha256, overlap_sha),
        (report.quantization_record_sha256, quantization_sha),
        (report.fp32_evaluation_sha256, fp32_sha),
        (report.int8_evaluation_sha256, int8_sha),
        (report.comparison_sha256, comparison_sha),
    )
    if any(declared != actual for declared, actual in links):
        raise ProductionRunError("Production report digest link mismatch")
    if (
        report.acceptance_passed != evidence.comparison.decision.overall_passed
        or report.calibration_requested_sample_count
        != evidence.pipeline.calibration.production_samples
        or report.calibration_actual_sample_count
        != quantization.calibration_sample_count
        or report.calibration_actual_sample_count
        != report.calibration_requested_sample_count
        or report.evaluation_requested_sample_count
        != evidence.pipeline.evaluation.production_samples
        or report.evaluation_actual_sample_count != fp32.selection.actual_sample_count
        or report.evaluation_used_complete_available_set
        != fp32.selection.used_complete_available_set
    ):
        raise ProductionRunError("Production report summary mismatch")

    reproducibility = evidence.reproducibility
    input_links = {
        "pipeline_config_sha256": pipeline_sha,
        "source_model_config_sha256": source_sha,
        "source_fp32_model_sha256": evidence.source.model.sha256,
        "calibration_manifest_sha256": calibration_sha,
        "evaluation_manifest_sha256": evaluation_sha,
    }
    output_links = {
        "source_inspection_sha256": inspection_sha,
        "calibration_validation_sha256": calibration_validation_sha,
        "evaluation_validation_sha256": evaluation_validation_sha,
        "overlap_report_sha256": overlap_sha,
        "quantization_record_sha256": quantization_sha,
        "int8_model_sha256": quantization.artifact.sha256,
        "fp32_evaluation_sha256": fp32_sha,
        "int8_evaluation_sha256": int8_sha,
        "comparison_sha256": comparison_sha,
        "production_report_sha256": report_sha,
    }
    if any(
        getattr(reproducibility.inputs, key) != value
        for key, value in input_links.items()
    ) or any(
        getattr(reproducibility.outputs, key) != value
        for key, value in output_links.items()
    ):
        raise ProductionRunError("Reproducibility digest graph mismatch")
    if (
        reproducibility.status != report.status
        or reproducibility.production_verified != report.production_verified
        or any(
            version != reproducibility.software.onnxruntime_version
            for version in (
                quantization.onnxruntime_version,
                fp32.onnxruntime_version,
                int8.onnxruntime_version,
            )
        )
        or reproducibility.execution.execution_provider != "CPUExecutionProvider"
    ):
        raise ProductionRunError("Reproducibility semantics mismatch")


def _validate_dataset_link(
    manifest: MobileNetV2P43BDatasetManifest,
    validation: MobileNetV2P43BDatasetValidationRecord,
) -> None:
    if manifest.dataset.purpose == "evaluation" and any(
        sample.label is None for sample in manifest.samples
    ):
        raise ProductionRunError("Formal evaluation data must be independently labelled")
    if (
        validation.dataset != manifest.dataset
        or validation.manifest_sha256 != sha256_canonical_json(manifest)
        or validation.sample_count != len(manifest.samples)
    ):
        raise ProductionRunError("Dataset validation record link mismatch")
    for declared, observed in zip(manifest.samples, validation.samples, strict=True):
        if (
            observed.id != declared.id
            or observed.path != declared.path
            or observed.label != declared.label
            or observed.declared_sha256 != declared.sha256
            or (
                declared.sha256 is not None
                and observed.observed_sha256 != declared.sha256
            )
        ):
            raise ProductionRunError("Dataset sample validation link mismatch")


def _record(
    records: dict[str, StrictModel],
    filename: str,
    model_type: type[Any],
) -> Any:
    value = records[f"records/{filename}"]
    if not isinstance(value, model_type):
        raise ProductionRunError(f"Unexpected record type: {filename}")
    return value


def _write_record(directory: Path, filename: str, record: StrictModel) -> None:
    write_canonical_json(directory / filename, record)


def _retain_failure(records: Path, stage: ProductionStage, exc: Exception) -> None:
    if not records.is_dir() or (records / "failure.json").exists():
        return
    completed = tuple(
        f"records/{path.name}"
        for path in sorted(records.glob("*.json"))
        if path.name != "failure.json"
    )
    raw_message = str(exc).replace("\x00", "\\0").strip() or type(exc).__name__
    message = raw_message[:2048].strip()
    failure = MobileNetV2P43BProductionFailureRecord(
        report_type="formal-production-validation",
        status="failed",
        production_verified=False,
        label_source="independent-ground-truth",
        failed_stage=stage,
        error_type=type(exc).__name__,
        message=message,
        completed_records=completed,
    )
    try:
        _write_record(records, "failure.json", failure)
    except (ModelPipelineError, OSError):
        return


def _safe_evidence_file(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    current = root
    for part in Path(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ProductionRunError(f"Symlinks are forbidden: {relative_path}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ProductionRunError(
            f"Unsafe or missing production evidence file: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise ProductionRunError(
            f"Production evidence is not a regular file: {relative_path}"
        )
    return resolved
