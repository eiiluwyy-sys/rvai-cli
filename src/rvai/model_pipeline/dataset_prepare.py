"""Deterministic class-balanced Manifest preparation for numeric ImageFolder data."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

from pydantic import NonNegativeInt, PositiveInt, ValidationError

from rvai.model_pipeline.dataset import require_no_dataset_overlap, validate_dataset
from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.io import (
    sha256_canonical_json,
    sha256_file,
    write_canonical_json,
)
from rvai.model_pipeline.schema import (
    Description,
    Identifier,
    MobileNetV2P43BDatasetIdentity,
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BDatasetSample,
    Sha256Digest,
    StrictModel,
)


class DatasetPreparationError(ModelPipelineError):
    """Raised when an ImageFolder cannot become safe formal Manifests."""


class MobileNetV2P43BDatasetPreparationRecord(StrictModel):
    """Portable selection evidence for two disjoint class-balanced Manifests."""

    schema_version: Literal["1.0"] = "1.0"
    preparation_type: Literal["numeric-imagefolder-class-balanced"]
    selection_policy: Literal["numeric-class-lexicographic-v1"]
    class_count: PositiveInt
    calibration_samples_per_class: PositiveInt
    evaluation_samples_per_class: PositiveInt
    source_sample_count: PositiveInt
    calibration_sample_count: PositiveInt
    evaluation_sample_count: PositiveInt
    unselected_sample_count: NonNegativeInt
    calibration_manifest_sha256: Sha256Digest
    evaluation_manifest_sha256: Sha256Digest
    overlap_report_sha256: Sha256Digest


@dataclass(frozen=True)
class PreparedDatasetManifests:
    """Generated strict Manifests plus deterministic preparation evidence."""

    calibration: MobileNetV2P43BDatasetManifest
    evaluation: MobileNetV2P43BDatasetManifest
    record: MobileNetV2P43BDatasetPreparationRecord


_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".ppm", ".webp"})


def prepare_numeric_imagefolder_manifests(
    dataset_root: Path | str,
    output_directory: Path | str,
    *,
    dataset_name: Identifier,
    dataset_version: Identifier,
    provenance: Description,
    license_description: Description,
    class_count: int = 1000,
    calibration_samples_per_class: int = 1,
    evaluation_samples_per_class: int = 5,
    pillow_image: ModuleType | None = None,
) -> PreparedDatasetManifests:
    """Prepare disjoint Manifests from directories named ``0`` through ``N-1``."""

    _require_positive("class_count", class_count)
    _require_positive(
        "calibration_samples_per_class",
        calibration_samples_per_class,
    )
    _require_positive(
        "evaluation_samples_per_class",
        evaluation_samples_per_class,
    )
    root = _validated_root(dataset_root)
    image_module = pillow_image or _load_pillow_image()
    expected_names = {str(label) for label in range(class_count)}
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise DatasetPreparationError(f"Cannot inspect dataset root: {exc}") from exc
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        additional = sorted(actual_names - expected_names)
        raise DatasetPreparationError(
            "Numeric class inventory mismatch; "
            f"missing={_inventory_summary(missing)}, "
            f"additional={_inventory_summary(additional)}"
        )

    calibration_samples: list[MobileNetV2P43BDatasetSample] = []
    evaluation_samples: list[MobileNetV2P43BDatasetSample] = []
    source_count = 0
    required_per_class = (
        calibration_samples_per_class + evaluation_samples_per_class
    )
    for label in range(class_count):
        class_directory = root / str(label)
        files = _validated_class_files(class_directory, image_module)
        source_count += len(files)
        if len(files) < required_per_class:
            raise DatasetPreparationError(
                f"Class {label} requires at least {required_per_class} images, "
                f"found {len(files)}"
            )
        calibration_files = files[:calibration_samples_per_class]
        evaluation_files = files[
            calibration_samples_per_class:required_per_class
        ]
        calibration_samples.extend(
            _sample(
                root,
                path,
                label,
                "calibration",
                index,
            )
            for index, path in enumerate(calibration_files)
        )
        evaluation_samples.extend(
            _sample(
                root,
                path,
                label,
                "evaluation",
                index,
            )
            for index, path in enumerate(evaluation_files)
        )

    calibration = _manifest(
        name=f"{dataset_name}-calibration",
        version=dataset_version,
        split="calibration",
        purpose="calibration",
        provenance=provenance,
        license_description=license_description,
        samples=tuple(calibration_samples),
    )
    evaluation = _manifest(
        name=f"{dataset_name}-evaluation",
        version=dataset_version,
        split="evaluation",
        purpose="evaluation",
        provenance=provenance,
        license_description=license_description,
        samples=tuple(evaluation_samples),
    )
    try:
        validated_calibration = validate_dataset(calibration, root)
        validated_evaluation = validate_dataset(evaluation, root)
        overlap = require_no_dataset_overlap(
            validated_calibration,
            validated_evaluation,
        )
    except ModelPipelineError as exc:
        raise DatasetPreparationError(
            f"Prepared dataset validation failed: {exc}"
        ) from exc
    selected_count = len(calibration_samples) + len(evaluation_samples)
    try:
        record = MobileNetV2P43BDatasetPreparationRecord(
            preparation_type="numeric-imagefolder-class-balanced",
            selection_policy="numeric-class-lexicographic-v1",
            class_count=class_count,
            calibration_samples_per_class=calibration_samples_per_class,
            evaluation_samples_per_class=evaluation_samples_per_class,
            source_sample_count=source_count,
            calibration_sample_count=len(calibration_samples),
            evaluation_sample_count=len(evaluation_samples),
            unselected_sample_count=source_count - selected_count,
            calibration_manifest_sha256=sha256_canonical_json(calibration),
            evaluation_manifest_sha256=sha256_canonical_json(evaluation),
            overlap_report_sha256=sha256_canonical_json(overlap),
        )
    except ValidationError as exc:
        raise DatasetPreparationError(
            f"Invalid dataset preparation record: {exc}"
        ) from exc

    output = Path(output_directory)
    try:
        output.mkdir()
    except FileExistsError as exc:
        raise DatasetPreparationError(
            f"Refusing to overwrite Manifest output: {output}"
        ) from exc
    except OSError as exc:
        raise DatasetPreparationError(f"Cannot create Manifest output: {exc}") from exc
    try:
        write_canonical_json(output / "calibration-manifest.yaml", calibration)
        write_canonical_json(output / "evaluation-manifest.yaml", evaluation)
        write_canonical_json(output / "preparation.json", record)
    except ModelPipelineError as exc:
        raise DatasetPreparationError(
            f"Cannot publish prepared Manifests: {exc}"
        ) from exc
    return PreparedDatasetManifests(
        calibration=calibration,
        evaluation=evaluation,
        record=record,
    )


def _validated_root(path: Path | str) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise DatasetPreparationError("Dataset root must not be a symlink")
    try:
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise DatasetPreparationError(f"Cannot resolve dataset root: {exc}") from exc
    if not root.is_dir():
        raise DatasetPreparationError("Dataset root must be a directory")
    return root


def _validated_class_files(directory: Path, pillow_image: ModuleType) -> tuple[Path, ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise DatasetPreparationError(
            f"Class entry must be a real directory: {directory.name}"
        )
    try:
        entries = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
    except OSError as exc:
        raise DatasetPreparationError(
            f"Cannot inspect class {directory.name}: {exc}"
        ) from exc
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise DatasetPreparationError(
                f"Class {directory.name} contains a non-regular entry: {path.name}"
            )
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise DatasetPreparationError(
                f"Class {directory.name} contains an unsupported file: {path.name}"
            )
        try:
            with pillow_image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise DatasetPreparationError(
                f"Cannot decode image {directory.name}/{path.name}: {exc}"
            ) from exc
    return entries


def _sample(
    root: Path,
    path: Path,
    label: int,
    purpose: Literal["calibration", "evaluation"],
    index: int,
) -> MobileNetV2P43BDatasetSample:
    relative = path.relative_to(root).as_posix()
    try:
        digest = sha256_file(path)
        return MobileNetV2P43BDatasetSample(
            id=f"{purpose}-{label:04d}-{index:04d}",
            path=relative,
            label=label,
            sha256=digest,
        )
    except ModelPipelineError as exc:
        raise DatasetPreparationError(
            f"Cannot identify selected image {relative}: {exc}"
        ) from exc
    except ValidationError as exc:
        raise DatasetPreparationError(
            f"Invalid selected image path or identity: {relative}: {exc}"
        ) from exc


def _manifest(
    *,
    name: str,
    version: str,
    split: str,
    purpose: Literal["calibration", "evaluation"],
    provenance: str,
    license_description: str,
    samples: tuple[MobileNetV2P43BDatasetSample, ...],
) -> MobileNetV2P43BDatasetManifest:
    try:
        return MobileNetV2P43BDatasetManifest(
            dataset=MobileNetV2P43BDatasetIdentity(
                name=name,
                version=version,
                split=split,
                purpose=purpose,
                provenance=provenance,
                license=license_description,
            ),
            preprocessing="mobilenet-v2-imagenet-v1",
            sample_order="manifest",
            samples=samples,
        )
    except ValidationError as exc:
        raise DatasetPreparationError(f"Invalid prepared Manifest: {exc}") from exc


def _load_pillow_image() -> ModuleType:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise DatasetPreparationError(
            'Pillow is required; install pip install -e ".[model-pipeline]"'
        ) from exc


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetPreparationError(f"{name} must be a positive integer")


def _inventory_summary(names: list[str]) -> str:
    if len(names) <= 10:
        return repr(names)
    return f"{names[:10]!r} ... ({len(names)} total)"
