"""Deterministic directory packaging and verification for P4.3B evidence."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import Field, PositiveInt, field_validator, model_validator

from rvai.model_pipeline.compare import (
    MobileNetV2P43BComparisonRecord,
    compare_evaluations,
)
from rvai.model_pipeline.dataset import (
    MobileNetV2P43BDatasetValidationRecord,
    MobileNetV2P43BOverlapReport,
)
from rvai.model_pipeline.environment import (
    MobileNetV2P43BReproducibilityRecord,
    MobileNetV2P43BSourceRevision,
    collect_source_revision,
)
from rvai.model_pipeline.errors import ModelPipelineError, PipelineIOError
from rvai.model_pipeline.evaluate import MobileNetV2P43BEvaluationRecord
from rvai.model_pipeline.inspect import MobileNetV2P43BSourceInspectionRecord
from rvai.model_pipeline.io import (
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_canonical_json,
    sha256_file,
    write_canonical_json,
)
from rvai.model_pipeline.pilot import MobileNetV2P43BProxyPilotReport
from rvai.model_pipeline.quantize import MobileNetV2P43BQuantizationRecord
from rvai.model_pipeline.report import render_comparison_markdown
from rvai.model_pipeline.schema import (
    CanonicalRelativePath,
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BPipelineConfig,
    MobileNetV2P43BSourceModelConfig,
    Sha256Digest,
    StrictModel,
)
from rvai.model_pipeline.synthetic import (
    MobileNetV2P43BPseudoLabelRecord,
    MobileNetV2P43BSyntheticGenerationRecord,
)


class EvidencePackageError(ModelPipelineError):
    """Raised when evidence cannot be validated, packaged, or published safely."""


PackageRole = Literal[
    "comparison-report",
    "int8-model",
    "pipeline-config",
    "source-model-config",
    "calibration-manifest",
    "unlabeled-evaluation-manifest",
    "evaluation-manifest",
    "generation-record",
    "pseudo-label-record",
    "source-inspection-record",
    "calibration-validation-record",
    "evaluation-validation-record",
    "overlap-report",
    "quantization-record",
    "fp32-evaluation-record",
    "int8-evaluation-record",
    "comparison-record",
    "proxy-pilot-report",
    "reproducibility-record",
]


_PAYLOAD_ROLES: dict[str, PackageRole] = {
    "comparison.md": "comparison-report",
    "models/mobilenetv2-12-int8.onnx": "int8-model",
    "records/calibration-manifest.json": "calibration-manifest",
    "records/calibration-validation.json": "calibration-validation-record",
    "records/comparison.json": "comparison-record",
    "records/evaluation-manifest.json": "evaluation-manifest",
    "records/evaluation-unlabeled-manifest.json": (
        "unlabeled-evaluation-manifest"
    ),
    "records/evaluation-validation.json": "evaluation-validation-record",
    "records/fp32-evaluation.json": "fp32-evaluation-record",
    "records/generation.json": "generation-record",
    "records/int8-evaluation.json": "int8-evaluation-record",
    "records/overlap.json": "overlap-report",
    "records/pipeline-config.json": "pipeline-config",
    "records/proxy-pilot-report.json": "proxy-pilot-report",
    "records/pseudo-labels.json": "pseudo-label-record",
    "records/quantization.json": "quantization-record",
    "records/reproducibility.json": "reproducibility-record",
    "records/source-inspection.json": "source-inspection-record",
    "records/source-model-config.json": "source-model-config",
}
PACKAGE_PAYLOAD_PATHS = tuple(sorted(_PAYLOAD_ROLES))


class MobileNetV2P43BPackageEntry(StrictModel):
    """Identity and role of one fixed package payload file."""

    path: CanonicalRelativePath
    role: PackageRole
    size_bytes: PositiveInt
    sha256: Sha256Digest

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


class MobileNetV2P43BPackageManifest(StrictModel):
    """Non-recursive identity of a provisional directory evidence package."""

    schema_version: Literal["1.0"] = "1.0"
    package_type: Literal["p43b-synthetic-proxy-evidence"]
    status: Literal["provisional"]
    production_verified: Literal[False]
    label_source: Literal["fp32-top1-pseudo-label"]
    run_source_revision: MobileNetV2P43BSourceRevision
    packager_source_revision: MobileNetV2P43BSourceRevision
    entry_count: Literal[19]
    entries: tuple[MobileNetV2P43BPackageEntry, ...] = Field(
        min_length=19,
        max_length=19,
    )
    content_sha256: Sha256Digest

    @field_validator("entries", mode="before")
    @classmethod
    def entry_list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("content_sha256")
    @classmethod
    def normalize_content_sha256(cls, value: str) -> str:
        return value.lower()

    @model_validator(mode="after")
    def validate_inventory(self) -> "MobileNetV2P43BPackageManifest":
        paths = tuple(entry.path for entry in self.entries)
        if paths != PACKAGE_PAYLOAD_PATHS:
            raise ValueError("package entries must match the fixed sorted inventory")
        if len(set(paths)) != len(paths):
            raise ValueError("package entry paths must be unique")
        if any(entry.role != _PAYLOAD_ROLES[entry.path] for entry in self.entries):
            raise ValueError("package entry role does not match its fixed path")
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count must equal the number of entries")
        if self.content_sha256 != package_content_sha256(self.entries):
            raise ValueError("content_sha256 does not match the ordered entries")
        return self


@dataclass(frozen=True)
class ValidatedPilotEvidence:
    """Strict portable records and the verified INT8 artifact path."""

    pipeline: MobileNetV2P43BPipelineConfig
    source: MobileNetV2P43BSourceModelConfig
    calibration_manifest: MobileNetV2P43BDatasetManifest
    unlabeled_evaluation_manifest: MobileNetV2P43BDatasetManifest
    evaluation_manifest: MobileNetV2P43BDatasetManifest
    generation: MobileNetV2P43BSyntheticGenerationRecord
    pseudo_labels: MobileNetV2P43BPseudoLabelRecord
    source_inspection: MobileNetV2P43BSourceInspectionRecord
    calibration_validation: MobileNetV2P43BDatasetValidationRecord
    evaluation_validation: MobileNetV2P43BDatasetValidationRecord
    overlap: MobileNetV2P43BOverlapReport
    quantization: MobileNetV2P43BQuantizationRecord
    fp32_evaluation: MobileNetV2P43BEvaluationRecord
    int8_evaluation: MobileNetV2P43BEvaluationRecord
    comparison: MobileNetV2P43BComparisonRecord
    proxy_report: MobileNetV2P43BProxyPilotReport
    reproducibility: MobileNetV2P43BReproducibilityRecord
    int8_model_path: Path


_RECORD_TYPES: dict[str, type[StrictModel]] = {
    "records/pipeline-config.json": MobileNetV2P43BPipelineConfig,
    "records/source-model-config.json": MobileNetV2P43BSourceModelConfig,
    "records/calibration-manifest.json": MobileNetV2P43BDatasetManifest,
    "records/evaluation-unlabeled-manifest.json": MobileNetV2P43BDatasetManifest,
    "records/evaluation-manifest.json": MobileNetV2P43BDatasetManifest,
    "records/generation.json": MobileNetV2P43BSyntheticGenerationRecord,
    "records/pseudo-labels.json": MobileNetV2P43BPseudoLabelRecord,
    "records/source-inspection.json": MobileNetV2P43BSourceInspectionRecord,
    "records/calibration-validation.json": MobileNetV2P43BDatasetValidationRecord,
    "records/evaluation-validation.json": MobileNetV2P43BDatasetValidationRecord,
    "records/overlap.json": MobileNetV2P43BOverlapReport,
    "records/quantization.json": MobileNetV2P43BQuantizationRecord,
    "records/fp32-evaluation.json": MobileNetV2P43BEvaluationRecord,
    "records/int8-evaluation.json": MobileNetV2P43BEvaluationRecord,
    "records/comparison.json": MobileNetV2P43BComparisonRecord,
    "records/proxy-pilot-report.json": MobileNetV2P43BProxyPilotReport,
    "records/reproducibility.json": MobileNetV2P43BReproducibilityRecord,
}


def package_content_sha256(
    entries: tuple[MobileNetV2P43BPackageEntry, ...],
) -> str:
    """Hash only the canonical JSON array of ordered package entries."""

    payload = [entry.model_dump(mode="json", exclude_none=False) for entry in entries]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        indent=None,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def validate_pilot_evidence(run_directory: Path | str) -> ValidatedPilotEvidence:
    """Strictly load and cross-validate one existing B4.1 run directory."""

    root = _validated_root(run_directory)
    records: dict[str, StrictModel] = {}
    for relative_path, model_type in _RECORD_TYPES.items():
        path = _safe_regular_file(root, relative_path)
        try:
            value = load_json(path, model_type)
        except PipelineIOError as exc:
            raise EvidencePackageError(f"Invalid evidence record {relative_path}: {exc}") from exc
        if path.read_bytes() != canonical_json_bytes(value):
            raise EvidencePackageError(
                f"Evidence record is not canonical JSON: {relative_path}"
            )
        records[relative_path] = value
    int8_model_path = _safe_regular_file(
        root,
        "models/mobilenetv2-12-int8.onnx",
    )
    evidence = ValidatedPilotEvidence(
        pipeline=_typed(records, "records/pipeline-config.json"),
        source=_typed(records, "records/source-model-config.json"),
        calibration_manifest=_typed(records, "records/calibration-manifest.json"),
        unlabeled_evaluation_manifest=_typed(
            records,
            "records/evaluation-unlabeled-manifest.json",
        ),
        evaluation_manifest=_typed(records, "records/evaluation-manifest.json"),
        generation=_typed(records, "records/generation.json"),
        pseudo_labels=_typed(records, "records/pseudo-labels.json"),
        source_inspection=_typed(records, "records/source-inspection.json"),
        calibration_validation=_typed(
            records,
            "records/calibration-validation.json",
        ),
        evaluation_validation=_typed(
            records,
            "records/evaluation-validation.json",
        ),
        overlap=_typed(records, "records/overlap.json"),
        quantization=_typed(records, "records/quantization.json"),
        fp32_evaluation=_typed(records, "records/fp32-evaluation.json"),
        int8_evaluation=_typed(records, "records/int8-evaluation.json"),
        comparison=_typed(records, "records/comparison.json"),
        proxy_report=_typed(records, "records/proxy-pilot-report.json"),
        reproducibility=_typed(records, "records/reproducibility.json"),
        int8_model_path=int8_model_path,
    )
    _validate_cross_record_graph(evidence)
    return evidence


def build_evidence_package(
    run_directory: Path | str,
    destination: Path | str,
    *,
    repository: Path | str | None = None,
    revision_provider: Callable[[Path | str], MobileNetV2P43BSourceRevision] = (
        collect_source_revision
    ),
) -> MobileNetV2P43BPackageManifest:
    """Build, verify, and atomically publish a new directory evidence package."""

    evidence = validate_pilot_evidence(run_directory)
    if not evidence.reproducibility.source_revision.working_tree_clean:
        raise EvidencePackageError("Run source revision has a dirty worktree")
    source_repository = (
        Path(__file__).resolve().parents[3] if repository is None else Path(repository)
    )
    try:
        packager_revision = revision_provider(source_repository)
    except ModelPipelineError as exc:
        raise EvidencePackageError(f"Cannot capture packager source revision: {exc}") from exc
    if not packager_revision.working_tree_clean:
        raise EvidencePackageError("Packager source revision has a dirty worktree")

    target = Path(destination)
    parent = target.parent
    if target.exists() or target.is_symlink():
        raise EvidencePackageError(f"Refusing to overwrite existing destination: {target}")
    if not parent.is_dir():
        raise EvidencePackageError(f"Package parent directory does not exist: {parent}")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".staging", dir=parent)
    )
    published = False
    try:
        (staging / "models").mkdir()
        (staging / "records").mkdir()
        report_bytes = render_comparison_markdown(
            evidence.comparison,
            evidence.quantization,
            evidence.proxy_report,
            evidence.reproducibility,
        ).encode("utf-8")
        _write_exclusive(staging / "comparison.md", report_bytes)
        source_root = _validated_root(run_directory)
        for relative_path in PACKAGE_PAYLOAD_PATHS:
            if relative_path == "comparison.md":
                continue
            source_path = _safe_regular_file(source_root, relative_path)
            _copy_exclusive(source_path, staging / relative_path)

        entries = tuple(
            MobileNetV2P43BPackageEntry(
                path=relative_path,
                role=_PAYLOAD_ROLES[relative_path],
                size_bytes=(staging / relative_path).stat().st_size,
                sha256=sha256_file(staging / relative_path),
            )
            for relative_path in PACKAGE_PAYLOAD_PATHS
        )
        manifest = MobileNetV2P43BPackageManifest(
            package_type="p43b-synthetic-proxy-evidence",
            status="provisional",
            production_verified=False,
            label_source="fp32-top1-pseudo-label",
            run_source_revision=evidence.reproducibility.source_revision,
            packager_source_revision=packager_revision,
            entry_count=19,
            entries=entries,
            content_sha256=package_content_sha256(entries),
        )
        write_canonical_json(staging / "package-manifest.json", manifest)
        _write_exclusive(staging / "sha256sums.txt", _sha256sums_bytes(entries))
        _fsync_tree(staging)
        verify_evidence_package(staging)
        _rename_directory_noreplace(staging, target)
        published = True
        _fsync_directory(parent)
        return manifest
    except EvidencePackageError:
        raise
    except (OSError, PipelineIOError) as exc:
        state = "published but not fully synchronized" if published else "not published"
        raise EvidencePackageError(
            f"Evidence package build failed ({state}): {exc}"
        ) from exc
    finally:
        if not published and staging.exists():
            _remove_staging(staging, parent, target.name)


def verify_evidence_package(
    package_directory: Path | str,
) -> MobileNetV2P43BPackageManifest:
    """Independently verify a complete package from disk without modifying it."""

    root = _validated_root(package_directory)
    expected_files = set(PACKAGE_PAYLOAD_PATHS) | {
        "package-manifest.json",
        "sha256sums.txt",
    }
    actual_files, actual_directories = _inventory_tree(root)
    if actual_directories != {"models", "records"}:
        raise EvidencePackageError("Package directory inventory is invalid")
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        additional = sorted(actual_files - expected_files)
        raise EvidencePackageError(
            f"Package file inventory mismatch; missing={missing}, additional={additional}"
        )

    manifest_path = _safe_regular_file(root, "package-manifest.json")
    try:
        manifest = load_json(manifest_path, MobileNetV2P43BPackageManifest)
    except PipelineIOError as exc:
        raise EvidencePackageError(f"Invalid package manifest: {exc}") from exc
    if manifest_path.read_bytes() != canonical_json_bytes(manifest):
        raise EvidencePackageError("package-manifest.json is not canonical JSON")
    if not manifest.run_source_revision.working_tree_clean:
        raise EvidencePackageError("Package records a dirty run source revision")
    if not manifest.packager_source_revision.working_tree_clean:
        raise EvidencePackageError("Package records a dirty packager source revision")

    for entry in manifest.entries:
        path = _safe_regular_file(root, entry.path)
        if path.stat().st_size != entry.size_bytes or sha256_file(path) != entry.sha256:
            raise EvidencePackageError(f"Package entry identity mismatch: {entry.path}")
    if manifest.content_sha256 != package_content_sha256(manifest.entries):
        raise EvidencePackageError("Package content_sha256 mismatch")
    sums_path = _safe_regular_file(root, "sha256sums.txt")
    if sums_path.read_bytes() != _sha256sums_bytes(manifest.entries):
        raise EvidencePackageError("sha256sums.txt does not match package entries")

    evidence = validate_pilot_evidence(root)
    expected_report = render_comparison_markdown(
        evidence.comparison,
        evidence.quantization,
        evidence.proxy_report,
        evidence.reproducibility,
    ).encode("utf-8")
    if _safe_regular_file(root, "comparison.md").read_bytes() != expected_report:
        raise EvidencePackageError("comparison.md does not match validated evidence")
    if manifest.run_source_revision != evidence.reproducibility.source_revision:
        raise EvidencePackageError("Manifest run revision does not match evidence")
    return manifest


def _validate_cross_record_graph(evidence: ValidatedPilotEvidence) -> None:
    pipeline_sha = sha256_canonical_json(evidence.pipeline)
    source_sha = sha256_canonical_json(evidence.source)
    calibration_sha = sha256_canonical_json(evidence.calibration_manifest)
    unlabeled_sha = sha256_canonical_json(evidence.unlabeled_evaluation_manifest)
    evaluation_sha = sha256_canonical_json(evidence.evaluation_manifest)
    inspection_sha = sha256_canonical_json(evidence.source_inspection)
    generation_sha = sha256_canonical_json(evidence.generation)
    pseudo_sha = sha256_canonical_json(evidence.pseudo_labels)
    calibration_validation_sha = sha256_canonical_json(
        evidence.calibration_validation
    )
    evaluation_validation_sha = sha256_canonical_json(evidence.evaluation_validation)
    overlap_sha = sha256_canonical_json(evidence.overlap)
    quantization_sha = sha256_canonical_json(evidence.quantization)
    fp32_sha = sha256_canonical_json(evidence.fp32_evaluation)
    int8_sha = sha256_canonical_json(evidence.int8_evaluation)
    comparison_sha = sha256_canonical_json(evidence.comparison)
    proxy_sha = sha256_canonical_json(evidence.proxy_report)

    if evidence.source_inspection.model != evidence.source.model:
        raise EvidencePackageError("Source inspection does not match source config")
    if evidence.generation.calibration_manifest_sha256 != calibration_sha:
        raise EvidencePackageError("Generation calibration manifest link mismatch")
    if evidence.generation.unlabeled_evaluation_manifest_sha256 != unlabeled_sha:
        raise EvidencePackageError("Generation evaluation manifest link mismatch")
    if evidence.pseudo_labels.unlabeled_manifest_sha256 != unlabeled_sha:
        raise EvidencePackageError("Pseudo-label input manifest link mismatch")
    if evidence.pseudo_labels.evaluation_manifest_sha256 != evaluation_sha:
        raise EvidencePackageError("Pseudo-label output manifest link mismatch")
    if evidence.pseudo_labels.source_model_sha256 != evidence.source.model.sha256:
        raise EvidencePackageError("Pseudo-label source model link mismatch")
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
        raise EvidencePackageError("Dataset overlap evidence is invalid")

    quantization = evidence.quantization
    if (
        quantization.source_model_sha256 != evidence.source.model.sha256
        or quantization.source_inspection_sha256 != inspection_sha
        or quantization.pipeline_config_sha256 != pipeline_sha
        or quantization.calibration_manifest_sha256 != calibration_sha
        or quantization.quantization != evidence.pipeline.quantization
    ):
        raise EvidencePackageError("Quantization record link mismatch")
    calibration_ids = tuple(
        sample.id
        for sample in evidence.calibration_validation.samples[
            : quantization.calibration_sample_count
        ]
    )
    if calibration_ids != quantization.calibration_sample_ids:
        raise EvidencePackageError("Quantization calibration order mismatch")
    if (
        evidence.int8_model_path.name != quantization.artifact.filename
        or evidence.int8_model_path.stat().st_size != quantization.artifact.size_bytes
        or sha256_file(evidence.int8_model_path) != quantization.artifact.sha256
    ):
        raise EvidencePackageError("INT8 model identity mismatch")

    fp32 = evidence.fp32_evaluation
    int8 = evidence.int8_evaluation
    if (
        fp32.model.role != "fp32"
        or fp32.model.filename != evidence.source.model.filename
        or fp32.model.size_bytes != evidence.source.model.size_bytes
        or fp32.model.sha256 != evidence.source.model.sha256
    ):
        raise EvidencePackageError("FP32 evaluation artifact mismatch")
    if (
        int8.model.role != "int8"
        or int8.model.filename != quantization.artifact.filename
        or int8.model.size_bytes != quantization.artifact.size_bytes
        or int8.model.sha256 != quantization.artifact.sha256
    ):
        raise EvidencePackageError("INT8 evaluation artifact mismatch")
    if (
        fp32.pipeline_config_sha256 != pipeline_sha
        or int8.pipeline_config_sha256 != pipeline_sha
        or fp32.evaluation_manifest_sha256 != evaluation_sha
        or int8.evaluation_manifest_sha256 != evaluation_sha
        or fp32.selection != int8.selection
    ):
        raise EvidencePackageError("Evaluation record link mismatch")
    expected_pairs = tuple(
        (sample.id, sample.label)
        for sample in evidence.evaluation_validation.samples[: fp32.selection.actual_sample_count]
    )
    for record in (fp32, int8):
        actual_pairs = tuple((sample.sample_id, sample.label) for sample in record.samples)
        if actual_pairs != expected_pairs:
            raise EvidencePackageError("Evaluation manifest order mismatch")

    recomputed = compare_evaluations(fp32, int8, evidence.pipeline.acceptance)
    if recomputed != evidence.comparison:
        raise EvidencePackageError("Comparison record does not recompute exactly")
    report = evidence.proxy_report
    report_links = (
        (report.pipeline_config_sha256, pipeline_sha),
        (report.source_inspection_sha256, inspection_sha),
        (report.generation_record_sha256, generation_sha),
        (report.pseudo_label_record_sha256, pseudo_sha),
        (report.overlap_report_sha256, overlap_sha),
        (report.quantization_record_sha256, quantization_sha),
        (report.fp32_evaluation_sha256, fp32_sha),
        (report.int8_evaluation_sha256, int8_sha),
        (report.comparison_sha256, comparison_sha),
    )
    if any(declared != actual for declared, actual in report_links):
        raise EvidencePackageError("Proxy report digest link mismatch")
    if (
        report.calibration_sample_count != evidence.generation.calibration_sample_count
        or report.evaluation_sample_count != evidence.generation.evaluation_sample_count
        or report.evaluation_sample_count != evidence.pseudo_labels.sample_count
        or report.proxy_acceptance_passed != evidence.comparison.decision.overall_passed
    ):
        raise EvidencePackageError("Proxy report summary mismatch")

    reproducibility = evidence.reproducibility
    input_links = {
        "pipeline_config_sha256": pipeline_sha,
        "source_model_config_sha256": source_sha,
        "source_fp32_model_sha256": evidence.source.model.sha256,
        "calibration_manifest_sha256": calibration_sha,
        "unlabeled_evaluation_manifest_sha256": unlabeled_sha,
        "evaluation_manifest_sha256": evaluation_sha,
    }
    output_links = {
        "source_inspection_sha256": inspection_sha,
        "generation_record_sha256": generation_sha,
        "pseudo_label_record_sha256": pseudo_sha,
        "calibration_validation_sha256": calibration_validation_sha,
        "evaluation_validation_sha256": evaluation_validation_sha,
        "overlap_report_sha256": overlap_sha,
        "quantization_record_sha256": quantization_sha,
        "int8_model_sha256": quantization.artifact.sha256,
        "fp32_evaluation_sha256": fp32_sha,
        "int8_evaluation_sha256": int8_sha,
        "comparison_sha256": comparison_sha,
        "proxy_pilot_report_sha256": proxy_sha,
    }
    if any(getattr(reproducibility.inputs, key) != value for key, value in input_links.items()):
        raise EvidencePackageError("Reproducibility input digest link mismatch")
    if any(
        getattr(reproducibility.outputs, key) != value
        for key, value in output_links.items()
    ):
        raise EvidencePackageError("Reproducibility output digest link mismatch")
    versions = reproducibility.software
    if any(
        value != versions.onnxruntime_version
        for value in (
            evidence.pseudo_labels.onnxruntime_version,
            quantization.onnxruntime_version,
            fp32.onnxruntime_version,
            int8.onnxruntime_version,
        )
    ):
        raise EvidencePackageError("ONNX Runtime version link mismatch")
    if any(
        provider != "CPUExecutionProvider"
        for provider in (
            evidence.pseudo_labels.execution_provider,
            quantization.execution_provider,
            fp32.execution_provider,
            int8.execution_provider,
            reproducibility.execution.execution_provider,
        )
    ):
        raise EvidencePackageError("Execution provider link mismatch")
    if (
        report.status != "provisional"
        or report.production_verified is not False
        or report.label_source != "fp32-top1-pseudo-label"
        or reproducibility.status != "provisional"
        or reproducibility.production_verified is not False
        or reproducibility.label_source != "fp32-top1-pseudo-label"
    ):
        raise EvidencePackageError("Evidence violates provisional semantics")
    render_comparison_markdown(
        evidence.comparison,
        evidence.quantization,
        report,
        reproducibility,
    )


def _validate_dataset_link(
    manifest: MobileNetV2P43BDatasetManifest,
    validation: MobileNetV2P43BDatasetValidationRecord,
) -> None:
    if (
        validation.dataset != manifest.dataset
        or validation.manifest_sha256 != sha256_canonical_json(manifest)
        or validation.sample_count != len(manifest.samples)
    ):
        raise EvidencePackageError("Dataset validation record link mismatch")
    for declared, observed in zip(manifest.samples, validation.samples, strict=True):
        if (
            observed.id != declared.id
            or observed.path != declared.path
            or observed.label != declared.label
            or observed.declared_sha256 != declared.sha256
            or declared.sha256 is None
            or observed.observed_sha256 != declared.sha256
        ):
            raise EvidencePackageError("Dataset sample validation link mismatch")


def _typed(records: dict[str, StrictModel], path: str) -> Any:
    return records[path]


def _validated_root(path: Path | str) -> Path:
    root = Path(path)
    if root.is_symlink():
        raise EvidencePackageError(f"Directory must not be a symlink: {root}")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise EvidencePackageError(f"Cannot resolve evidence directory: {exc}") from exc
    if not resolved.is_dir():
        raise EvidencePackageError(f"Evidence root is not a directory: {root}")
    return resolved


def _safe_regular_file(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    current = root
    for part in Path(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise EvidencePackageError(f"Symlinks are forbidden: {relative_path}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidencePackageError(f"Unsafe or missing package file: {relative_path}") from exc
    if not resolved.is_file():
        raise EvidencePackageError(f"Package payload is not a regular file: {relative_path}")
    return resolved


def _inventory_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        relative_base = base.relative_to(root)
        for name in names:
            path = base / name
            relative = (relative_base / name).as_posix()
            if path.is_symlink():
                raise EvidencePackageError(f"Symlinks are forbidden: {relative}")
            directories.add(relative)
        for name in filenames:
            path = base / name
            relative = (relative_base / name).as_posix()
            if path.is_symlink():
                raise EvidencePackageError(f"Symlinks are forbidden: {relative}")
            if not path.is_file():
                raise EvidencePackageError(f"Non-regular package file: {relative}")
            files.add(relative)
    return files, directories


def _sha256sums_bytes(entries: tuple[MobileNetV2P43BPackageEntry, ...]) -> bytes:
    return "".join(f"{entry.sha256}  {entry.path}\n" for entry in entries).encode(
        "utf-8"
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise EvidencePackageError(f"Cannot write staged package file: {path.name}") from exc


def _copy_exclusive(source: Path, destination: Path) -> None:
    try:
        with source.open("rb") as input_handle, destination.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except OSError as exc:
        raise EvidencePackageError(
            f"Cannot copy staged package file: {destination.name}"
        ) from exc


def _fsync_tree(root: Path) -> None:
    for relative in ("models", "records"):
        _fsync_directory(root / relative)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise EvidencePackageError("Atomic no-replace directory publication is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise EvidencePackageError(
                f"Refusing to overwrite existing destination: {destination}"
            )
        if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise EvidencePackageError(
                "Atomic no-replace directory publication is unavailable"
            )
        raise EvidencePackageError(
            f"Cannot publish evidence package: {os.strerror(error)}"
        )


def _remove_staging(staging: Path, parent: Path, destination_name: str) -> None:
    expected_prefix = f".{destination_name}."
    if staging.parent != parent or not staging.name.startswith(expected_prefix):
        raise EvidencePackageError("Refusing unsafe staging cleanup")
    shutil.rmtree(staging)
