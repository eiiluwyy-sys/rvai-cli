"""External dataset validation and overlap detection for P4.3B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import NonNegativeInt, PositiveInt, model_validator

from rvai.model_pipeline.errors import ModelPipelineError, PipelineIOError
from rvai.model_pipeline.io import sha256_canonical_json, sha256_file
from rvai.model_pipeline.schema import (
    CanonicalRelativePath,
    Identifier,
    MobileNetV2P43BDatasetIdentity,
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BDatasetSample,
    Sha256Digest,
    StrictModel,
)


class DatasetValidationError(ModelPipelineError):
    """Raised when an external dataset root or sample is unsafe or invalid."""


class DatasetOverlapError(ModelPipelineError):
    """Raised when calibration and evaluation selections overlap."""


class MobileNetV2P43BValidatedSampleRecord(StrictModel):
    """Deterministic evidence for one validated external sample."""

    id: Identifier
    path: CanonicalRelativePath
    label: NonNegativeInt | None
    declared_sha256: Sha256Digest | None
    observed_sha256: Sha256Digest
    size_bytes: NonNegativeInt


class MobileNetV2P43BDatasetValidationRecord(StrictModel):
    """Deterministic dataset validation evidence without host paths."""

    schema_version: Literal["1.0"] = "1.0"
    dataset: MobileNetV2P43BDatasetIdentity
    manifest_sha256: Sha256Digest
    sample_order: Literal["manifest"]
    sample_count: PositiveInt
    samples: tuple[MobileNetV2P43BValidatedSampleRecord, ...]

    @model_validator(mode="after")
    def count_matches_samples(self) -> "MobileNetV2P43BDatasetValidationRecord":
        if self.sample_count != len(self.samples):
            raise ValueError("sample_count must equal the number of samples")
        return self


class MobileNetV2P43BOverlapPair(StrictModel):
    """One calibration/evaluation pair with one or more overlap reasons."""

    calibration_id: Identifier
    evaluation_id: Identifier
    reasons: tuple[Literal["sample_id", "content_sha256", "resolved_file"], ...]


class MobileNetV2P43BOverlapReport(StrictModel):
    """Deterministic overlap decision for two validated manifests."""

    schema_version: Literal["1.0"] = "1.0"
    calibration_manifest_sha256: Sha256Digest
    evaluation_manifest_sha256: Sha256Digest
    overlap_count: NonNegativeInt
    overlaps: tuple[MobileNetV2P43BOverlapPair, ...]

    @model_validator(mode="after")
    def count_matches_pairs(self) -> "MobileNetV2P43BOverlapReport":
        if self.overlap_count != len(self.overlaps):
            raise ValueError("overlap_count must equal the number of overlap pairs")
        return self


@dataclass(frozen=True)
class ResolvedDatasetSample:
    """Runtime-only sample path paired with deterministic evidence."""

    declaration: MobileNetV2P43BDatasetSample
    resolved_path: Path
    record: MobileNetV2P43BValidatedSampleRecord


@dataclass(frozen=True)
class ValidatedDataset:
    """Validated runtime paths plus the portable deterministic record."""

    manifest: MobileNetV2P43BDatasetManifest
    root: Path
    samples: tuple[ResolvedDatasetSample, ...]
    record: MobileNetV2P43BDatasetValidationRecord


def validate_dataset(
    manifest: MobileNetV2P43BDatasetManifest,
    root: Path | str,
) -> ValidatedDataset:
    """Resolve and verify every manifest sample while preserving its order."""

    root_path = _resolve_root(root)
    resolved_samples: list[ResolvedDatasetSample] = []
    for sample in manifest.samples:
        resolved_path = _resolve_sample(root_path, sample)
        try:
            stat_result = resolved_path.stat()
            observed_sha256 = sha256_file(resolved_path)
        except (OSError, PipelineIOError) as exc:
            raise DatasetValidationError(
                f"Cannot validate dataset sample {sample.id}: {exc}"
            ) from exc
        if sample.sha256 is not None and observed_sha256 != sample.sha256:
            raise DatasetValidationError(
                f"Dataset sample {sample.id} SHA-256 mismatch: "
                f"expected {sample.sha256}, got {observed_sha256}"
            )
        if manifest.dataset.purpose == "evaluation":
            if sample.label is None or sample.label > 999:
                raise DatasetValidationError(
                    f"Evaluation sample {sample.id} label must be in range 0 through 999"
                )
        sample_record = MobileNetV2P43BValidatedSampleRecord(
            id=sample.id,
            path=sample.path,
            label=sample.label,
            declared_sha256=sample.sha256,
            observed_sha256=observed_sha256,
            size_bytes=stat_result.st_size,
        )
        resolved_samples.append(
            ResolvedDatasetSample(
                declaration=sample,
                resolved_path=resolved_path,
                record=sample_record,
            )
        )

    records = tuple(sample.record for sample in resolved_samples)
    record = MobileNetV2P43BDatasetValidationRecord(
        dataset=manifest.dataset,
        manifest_sha256=sha256_canonical_json(manifest),
        sample_order="manifest",
        sample_count=len(records),
        samples=records,
    )
    return ValidatedDataset(
        manifest=manifest,
        root=root_path,
        samples=tuple(resolved_samples),
        record=record,
    )


def detect_dataset_overlap(
    calibration: ValidatedDataset,
    evaluation: ValidatedDataset,
) -> MobileNetV2P43BOverlapReport:
    """Compare stable IDs, content digests, and resolved file identities."""

    if calibration.manifest.dataset.purpose != "calibration":
        raise DatasetOverlapError("First dataset must have calibration purpose")
    if evaluation.manifest.dataset.purpose != "evaluation":
        raise DatasetOverlapError("Second dataset must have evaluation purpose")

    pairs: list[MobileNetV2P43BOverlapPair] = []
    for calibration_sample in calibration.samples:
        for evaluation_sample in evaluation.samples:
            reasons: list[str] = []
            if calibration_sample.declaration.id == evaluation_sample.declaration.id:
                reasons.append("sample_id")
            if (
                calibration_sample.record.observed_sha256
                == evaluation_sample.record.observed_sha256
            ):
                reasons.append("content_sha256")
            if calibration_sample.resolved_path == evaluation_sample.resolved_path:
                reasons.append("resolved_file")
            if reasons:
                pairs.append(
                    MobileNetV2P43BOverlapPair(
                        calibration_id=calibration_sample.declaration.id,
                        evaluation_id=evaluation_sample.declaration.id,
                        reasons=tuple(reasons),
                    )
                )

    return MobileNetV2P43BOverlapReport(
        calibration_manifest_sha256=calibration.record.manifest_sha256,
        evaluation_manifest_sha256=evaluation.record.manifest_sha256,
        overlap_count=len(pairs),
        overlaps=tuple(pairs),
    )


def require_no_dataset_overlap(
    calibration: ValidatedDataset,
    evaluation: ValidatedDataset,
) -> MobileNetV2P43BOverlapReport:
    """Return zero-overlap evidence or fail before calibration begins."""

    report = detect_dataset_overlap(calibration, evaluation)
    if report.overlap_count:
        raise DatasetOverlapError(
            "Calibration and evaluation datasets overlap in "
            f"{report.overlap_count} sample pair(s)"
        )
    return report


def _resolve_root(root: Path | str) -> Path:
    requested = Path(root)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise DatasetValidationError(f"Cannot resolve dataset root: {detail}") from exc
    if not resolved.is_dir():
        raise DatasetValidationError("Dataset root must be a directory")
    return resolved


def _resolve_sample(
    root: Path,
    sample: MobileNetV2P43BDatasetSample,
) -> Path:
    try:
        resolved = (root / sample.path).resolve(strict=True)
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise DatasetValidationError(
            f"Cannot resolve dataset sample {sample.id}: {detail}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise DatasetValidationError(
            f"Dataset sample {sample.id} escapes the supplied root"
        )
    if not resolved.is_file():
        raise DatasetValidationError(
            f"Dataset sample {sample.id} must resolve to a regular file"
        )
    return resolved
