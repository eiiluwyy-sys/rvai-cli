import hashlib
from pathlib import Path
import shutil

import pytest

from rvai.model_pipeline.dataset import (
    DatasetOverlapError,
    DatasetValidationError,
    detect_dataset_overlap,
    require_no_dataset_overlap,
    validate_dataset,
)
from rvai.model_pipeline.io import canonical_json_text
from rvai.model_pipeline.schema import MobileNetV2P43BDatasetManifest


COMMITTED_IMAGE = Path(__file__).parent / "fixtures" / "onnx" / "red-image.ppm"


def manifest(
    purpose: str,
    samples: list[dict[str, object]],
) -> MobileNetV2P43BDatasetManifest:
    return MobileNetV2P43BDatasetManifest.model_validate(
        {
            "schema_version": "1.0",
            "dataset": {
                "name": f"{purpose}-dataset",
                "version": "v1",
                "split": purpose,
                "purpose": purpose,
                "provenance": "Tiny committed offline fixture.",
                "license": "Test fixture only.",
            },
            "preprocessing": "mobilenet-v2-imagenet-v1",
            "sample_order": "manifest",
            "samples": samples,
        }
    )


def copy_fixture(root: Path, relative_path: str, *, contents: bytes | None = None) -> Path:
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if contents is None:
        shutil.copy(COMMITTED_IMAGE, destination)
    else:
        destination.write_bytes(contents)
    return destination


def test_dataset_validation_preserves_order_and_records_portable_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    first = copy_fixture(root, "class-000/first.ppm")
    second = copy_fixture(root, "class-001/second.ppm", contents=b"second-image")
    first_sha256 = hashlib.sha256(first.read_bytes()).hexdigest()
    declared = manifest(
        "evaluation",
        [
            {"id": "sample-a", "path": "class-000/first.ppm", "label": 0},
            {
                "id": "sample-b",
                "path": "class-001/second.ppm",
                "label": 999,
                "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
            },
        ],
    )

    validated = validate_dataset(declared, root)

    assert [sample.declaration.id for sample in validated.samples] == [
        "sample-a",
        "sample-b",
    ]
    assert validated.samples[0].resolved_path == first.resolve()
    assert validated.record.samples[0].observed_sha256 == first_sha256
    assert validated.record.samples[1].declared_sha256 is not None
    assert validated.record.sample_count == 2
    serialized = canonical_json_text(validated.record)
    assert str(root.resolve()) not in serialized
    assert "class-000/first.ppm" in serialized


def test_calibration_labels_may_be_omitted(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    copy_fixture(root, "calibration.ppm")
    declared = manifest(
        "calibration",
        [{"id": "calibration-a", "path": "calibration.ppm"}],
    )
    validated = validate_dataset(declared, root)
    assert validated.record.samples[0].label is None


def test_evaluation_label_must_not_exceed_imagenet_range(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    copy_fixture(root, "evaluation.ppm")
    declared = manifest(
        "evaluation",
        [{"id": "evaluation-a", "path": "evaluation.ppm", "label": 1000}],
    )
    with pytest.raises(DatasetValidationError, match="0 through 999"):
        validate_dataset(declared, root)


def test_optional_sample_digest_is_verified(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    copy_fixture(root, "sample.ppm")
    declared = manifest(
        "calibration",
        [
            {
                "id": "calibration-a",
                "path": "sample.ppm",
                "sha256": "0" * 64,
            }
        ],
    )
    with pytest.raises(DatasetValidationError, match="SHA-256 mismatch"):
        validate_dataset(declared, root)


def test_sample_symlink_cannot_escape_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    external = tmp_path / "external.ppm"
    shutil.copy(COMMITTED_IMAGE, external)
    (root / "escape.ppm").symlink_to(external)
    declared = manifest(
        "calibration",
        [{"id": "calibration-a", "path": "escape.ppm"}],
    )
    with pytest.raises(DatasetValidationError, match="escapes"):
        validate_dataset(declared, root)


def test_sample_must_resolve_to_regular_file(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "directory").mkdir(parents=True)
    declared = manifest(
        "calibration",
        [{"id": "calibration-a", "path": "directory"}],
    )
    with pytest.raises(DatasetValidationError, match="regular file"):
        validate_dataset(declared, root)


def test_missing_sample_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    declared = manifest(
        "calibration",
        [{"id": "calibration-a", "path": "missing.ppm"}],
    )
    with pytest.raises(DatasetValidationError, match="Cannot resolve"):
        validate_dataset(declared, root)


def test_overlap_detection_uses_ids_content_and_resolved_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    shared = copy_fixture(root, "shared.ppm")
    calibration = validate_dataset(
        manifest(
            "calibration",
            [{"id": "shared-id", "path": "shared.ppm"}],
        ),
        root,
    )
    evaluation = validate_dataset(
        manifest(
            "evaluation",
            [{"id": "shared-id", "path": "shared.ppm", "label": 0}],
        ),
        root,
    )
    assert shared.is_file()

    report = detect_dataset_overlap(calibration, evaluation)

    assert report.overlap_count == 1
    assert report.overlaps[0].reasons == (
        "sample_id",
        "content_sha256",
        "resolved_file",
    )
    with pytest.raises(DatasetOverlapError, match="1 sample pair"):
        require_no_dataset_overlap(calibration, evaluation)


def test_overlap_detection_finds_same_content_under_different_names(
    tmp_path: Path,
) -> None:
    calibration_root = tmp_path / "calibration"
    evaluation_root = tmp_path / "evaluation"
    copy_fixture(calibration_root, "calibration.ppm")
    copy_fixture(evaluation_root, "evaluation.ppm")
    calibration = validate_dataset(
        manifest(
            "calibration",
            [{"id": "calibration-a", "path": "calibration.ppm"}],
        ),
        calibration_root,
    )
    evaluation = validate_dataset(
        manifest(
            "evaluation",
            [{"id": "evaluation-a", "path": "evaluation.ppm", "label": 0}],
        ),
        evaluation_root,
    )
    report = detect_dataset_overlap(calibration, evaluation)
    assert report.overlaps[0].reasons == ("content_sha256",)


def test_non_overlapping_datasets_produce_zero_overlap_record(tmp_path: Path) -> None:
    calibration_root = tmp_path / "calibration"
    evaluation_root = tmp_path / "evaluation"
    copy_fixture(calibration_root, "calibration.ppm", contents=b"calibration")
    copy_fixture(evaluation_root, "evaluation.ppm", contents=b"evaluation")
    calibration = validate_dataset(
        manifest(
            "calibration",
            [{"id": "calibration-a", "path": "calibration.ppm"}],
        ),
        calibration_root,
    )
    evaluation = validate_dataset(
        manifest(
            "evaluation",
            [{"id": "evaluation-a", "path": "evaluation.ppm", "label": 0}],
        ),
        evaluation_root,
    )
    report = require_no_dataset_overlap(calibration, evaluation)
    assert report.overlap_count == 0
    assert report.overlaps == ()
