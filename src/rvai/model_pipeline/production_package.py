"""Deterministic packaging for formal independently labelled P4.3B evidence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PositiveInt, field_validator, model_validator

from rvai.model_pipeline.environment import (
    MobileNetV2P43BSourceRevision,
    collect_source_revision,
)
from rvai.model_pipeline.errors import ModelPipelineError, PipelineIOError
from rvai.model_pipeline.io import (
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from rvai.model_pipeline.package import (
    _copy_exclusive,
    _fsync_directory,
    _fsync_tree,
    _inventory_tree,
    _remove_staging,
    _rename_directory_noreplace,
    _safe_regular_file,
    _validated_root,
    _write_exclusive,
)
from rvai.model_pipeline.production import validate_production_evidence
from rvai.model_pipeline.production_report import render_production_markdown
from rvai.model_pipeline.schema import (
    CanonicalRelativePath,
    Sha256Digest,
    StrictModel,
)


class ProductionPackageError(ModelPipelineError):
    """Raised when formal evidence cannot be packaged or verified safely."""


ProductionPackageRole = Literal[
    "comparison-report",
    "int8-model",
    "pipeline-config",
    "source-model-config",
    "calibration-manifest",
    "evaluation-manifest",
    "source-inspection-record",
    "calibration-validation-record",
    "evaluation-validation-record",
    "overlap-report",
    "quantization-record",
    "fp32-evaluation-record",
    "int8-evaluation-record",
    "comparison-record",
    "production-report",
    "reproducibility-record",
]


_PRODUCTION_PAYLOAD_ROLES: dict[str, ProductionPackageRole] = {
    "comparison.md": "comparison-report",
    "models/mobilenetv2-12-int8.onnx": "int8-model",
    "records/calibration-manifest.json": "calibration-manifest",
    "records/calibration-validation.json": "calibration-validation-record",
    "records/comparison.json": "comparison-record",
    "records/evaluation-manifest.json": "evaluation-manifest",
    "records/evaluation-validation.json": "evaluation-validation-record",
    "records/fp32-evaluation.json": "fp32-evaluation-record",
    "records/int8-evaluation.json": "int8-evaluation-record",
    "records/overlap.json": "overlap-report",
    "records/pipeline-config.json": "pipeline-config",
    "records/production-report.json": "production-report",
    "records/quantization.json": "quantization-record",
    "records/reproducibility.json": "reproducibility-record",
    "records/source-inspection.json": "source-inspection-record",
    "records/source-model-config.json": "source-model-config",
}
PRODUCTION_PACKAGE_PAYLOAD_PATHS = tuple(sorted(_PRODUCTION_PAYLOAD_ROLES))


class MobileNetV2P43BProductionPackageEntry(StrictModel):
    """Identity and role of one formal package payload file."""

    path: CanonicalRelativePath
    role: ProductionPackageRole
    size_bytes: PositiveInt
    sha256: Sha256Digest

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


class MobileNetV2P43BProductionPackageManifest(StrictModel):
    """Non-recursive identity of a formal accepted or rejected package."""

    schema_version: Literal["1.0"] = "1.0"
    package_type: Literal["p43b-production-evidence"]
    status: Literal["accepted", "rejected"]
    production_verified: bool
    label_source: Literal["independent-ground-truth"]
    run_source_revision: MobileNetV2P43BSourceRevision
    packager_source_revision: MobileNetV2P43BSourceRevision
    entry_count: Literal[16]
    entries: tuple[MobileNetV2P43BProductionPackageEntry, ...] = Field(
        min_length=16,
        max_length=16,
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
    def validate_inventory(
        self,
    ) -> "MobileNetV2P43BProductionPackageManifest":
        paths = tuple(entry.path for entry in self.entries)
        if paths != PRODUCTION_PACKAGE_PAYLOAD_PATHS:
            raise ValueError("package entries must match the fixed sorted inventory")
        if any(
            entry.role != _PRODUCTION_PAYLOAD_ROLES[entry.path]
            for entry in self.entries
        ):
            raise ValueError("package entry role does not match its fixed path")
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count must equal the number of entries")
        if self.content_sha256 != production_package_content_sha256(self.entries):
            raise ValueError("content_sha256 does not match ordered entries")
        if (self.status == "accepted") != self.production_verified:
            raise ValueError("status must match production_verified")
        if (
            not self.run_source_revision.working_tree_clean
            or not self.packager_source_revision.working_tree_clean
        ):
            raise ValueError("formal packages require clean source revisions")
        return self


def production_package_content_sha256(
    entries: tuple[MobileNetV2P43BProductionPackageEntry, ...],
) -> str:
    """Hash the canonical ordered package-entry array only."""

    payload = [entry.model_dump(mode="json", exclude_none=False) for entry in entries]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def build_production_evidence_package(
    run_directory: Path | str,
    output_directory: Path | str,
    *,
    repository: Path | str,
) -> MobileNetV2P43BProductionPackageManifest:
    """Validate, stage, verify, and atomically publish a formal package."""

    try:
        evidence = validate_production_evidence(run_directory)
    except ModelPipelineError as exc:
        raise ProductionPackageError(f"Invalid formal run evidence: {exc}") from exc
    packager_revision = collect_source_revision(repository)
    if not packager_revision.working_tree_clean:
        raise ProductionPackageError("Formal packaging requires a clean Git tree")
    if not evidence.reproducibility.source_revision.working_tree_clean:
        raise ProductionPackageError("Run evidence records a dirty Git tree")

    target = Path(output_directory)
    parent = target.parent.resolve(strict=True)
    if target.exists() or target.is_symlink():
        raise ProductionPackageError(
            f"Refusing to overwrite existing destination: {target}"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    )
    published = False
    try:
        (staging / "models").mkdir()
        (staging / "records").mkdir()
        markdown = render_production_markdown(
            evidence.comparison,
            evidence.quantization,
            evidence.report,
            evidence.reproducibility,
        ).encode("utf-8")
        _write_exclusive(staging / "comparison.md", markdown)
        source_root = _validated_root(run_directory)
        for relative_path in PRODUCTION_PACKAGE_PAYLOAD_PATHS:
            if relative_path == "comparison.md":
                continue
            _copy_exclusive(
                _safe_regular_file(source_root, relative_path),
                staging / relative_path,
            )
        entries = tuple(
            MobileNetV2P43BProductionPackageEntry(
                path=relative_path,
                role=_PRODUCTION_PAYLOAD_ROLES[relative_path],
                size_bytes=(staging / relative_path).stat().st_size,
                sha256=sha256_file(staging / relative_path),
            )
            for relative_path in PRODUCTION_PACKAGE_PAYLOAD_PATHS
        )
        manifest = MobileNetV2P43BProductionPackageManifest(
            package_type="p43b-production-evidence",
            status=evidence.report.status,
            production_verified=evidence.report.production_verified,
            label_source="independent-ground-truth",
            run_source_revision=evidence.reproducibility.source_revision,
            packager_source_revision=packager_revision,
            entry_count=16,
            entries=entries,
            content_sha256=production_package_content_sha256(entries),
        )
        write_canonical_json(staging / "package-manifest.json", manifest)
        _write_exclusive(staging / "sha256sums.txt", _sha256sums_bytes(entries))
        _fsync_tree(staging)
        verify_production_evidence_package(staging)
        _rename_directory_noreplace(staging, target)
        published = True
        _fsync_directory(parent)
        return manifest
    except ProductionPackageError:
        raise
    except (ModelPipelineError, OSError, PipelineIOError, ValueError) as exc:
        state = "published but not synchronized" if published else "not published"
        raise ProductionPackageError(
            f"Formal package build failed ({state}): {exc}"
        ) from exc
    finally:
        if not published and staging.exists():
            _remove_staging(staging, parent, target.name)


def verify_production_evidence_package(
    package_directory: Path | str,
) -> MobileNetV2P43BProductionPackageManifest:
    """Independently verify one formal package without modifying it."""

    root = _validated_root(package_directory)
    expected_files = set(PRODUCTION_PACKAGE_PAYLOAD_PATHS) | {
        "package-manifest.json",
        "sha256sums.txt",
    }
    actual_files, actual_directories = _inventory_tree(root)
    if actual_directories != {"models", "records"}:
        raise ProductionPackageError("Package directory inventory is invalid")
    if actual_files != expected_files:
        raise ProductionPackageError(
            "Package file inventory mismatch; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"additional={sorted(actual_files - expected_files)}"
        )
    manifest_path = _safe_regular_file(root, "package-manifest.json")
    try:
        manifest = load_json(
            manifest_path,
            MobileNetV2P43BProductionPackageManifest,
        )
    except PipelineIOError as exc:
        raise ProductionPackageError(f"Invalid package manifest: {exc}") from exc
    if manifest_path.read_bytes() != canonical_json_bytes(manifest):
        raise ProductionPackageError("package-manifest.json is not canonical JSON")
    for entry in manifest.entries:
        path = _safe_regular_file(root, entry.path)
        if path.stat().st_size != entry.size_bytes or sha256_file(path) != entry.sha256:
            raise ProductionPackageError(
                f"Package entry identity mismatch: {entry.path}"
            )
    if manifest.content_sha256 != production_package_content_sha256(manifest.entries):
        raise ProductionPackageError("Package content digest mismatch")
    if _safe_regular_file(root, "sha256sums.txt").read_bytes() != _sha256sums_bytes(
        manifest.entries
    ):
        raise ProductionPackageError("sha256sums.txt does not match package entries")
    try:
        evidence = validate_production_evidence(root)
    except ModelPipelineError as exc:
        raise ProductionPackageError(f"Invalid packaged evidence: {exc}") from exc
    expected_report = render_production_markdown(
        evidence.comparison,
        evidence.quantization,
        evidence.report,
        evidence.reproducibility,
    ).encode("utf-8")
    if _safe_regular_file(root, "comparison.md").read_bytes() != expected_report:
        raise ProductionPackageError("comparison.md does not match evidence")
    if (
        manifest.run_source_revision != evidence.reproducibility.source_revision
        or manifest.status != evidence.report.status
        or manifest.production_verified != evidence.report.production_verified
    ):
        raise ProductionPackageError("Package manifest semantics mismatch")
    return manifest


def _sha256sums_bytes(
    entries: tuple[MobileNetV2P43BProductionPackageEntry, ...],
) -> bytes:
    return "".join(
        f"{entry.sha256}  {entry.path}\n" for entry in entries
    ).encode("utf-8")
