import importlib.util
from pathlib import Path

import pytest

from rvai.model_pipeline.dataset_prepare import (
    DatasetPreparationError,
    MobileNetV2P43BDatasetPreparationRecord,
    prepare_numeric_imagefolder_manifests,
)
from rvai.model_pipeline.io import load_json, sha256_file
from rvai.model_pipeline.schema import MobileNetV2P43BDatasetManifest

HAS_PILLOW = importlib.util.find_spec("PIL") is not None
pytestmark = pytest.mark.skipif(not HAS_PILLOW, reason="Pillow is not installed")


def _write_ppm(path: Path, rgb: tuple[int, int, int]) -> None:
    path.write_bytes(b"P6\n8 8\n255\n" + bytes(rgb) * 64)


def _imagefolder(root: Path, *, duplicate: bool = False) -> None:
    root.mkdir()
    for label in range(2):
        directory = root / str(label)
        directory.mkdir()
        for index in range(4):
            color = (label * 50 + index, index + 10, label + 20)
            if duplicate and label == 0 and index == 1:
                color = (0, 10, 20)
            _write_ppm(directory / f"image-{index}.ppm", color)


def test_prepare_numeric_imagefolder_is_balanced_and_deterministic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    _imagefolder(root)
    output = tmp_path / "manifests"

    prepared = prepare_numeric_imagefolder_manifests(
        root,
        output,
        dataset_name="imagenetv2-fixture",
        dataset_version="fixture-v1",
        provenance="Independently labelled deterministic test fixture.",
        license_description="Generated fixture for repository tests.",
        class_count=2,
        calibration_samples_per_class=1,
        evaluation_samples_per_class=2,
    )

    assert prepared.record.source_sample_count == 8
    assert prepared.record.calibration_sample_count == 2
    assert prepared.record.evaluation_sample_count == 4
    assert prepared.record.unselected_sample_count == 2
    assert tuple(sample.label for sample in prepared.calibration.samples) == (0, 1)
    assert tuple(sample.label for sample in prepared.evaluation.samples) == (
        0,
        0,
        1,
        1,
    )
    assert prepared.calibration.samples[0].path == "0/image-0.ppm"
    assert prepared.evaluation.samples[0].path == "0/image-1.ppm"
    for sample in (*prepared.calibration.samples, *prepared.evaluation.samples):
        assert sample.sha256 == sha256_file(root / sample.path)
    assert load_json(
        output / "calibration-manifest.yaml",
        MobileNetV2P43BDatasetManifest,
    ) == prepared.calibration
    assert load_json(
        output / "evaluation-manifest.yaml",
        MobileNetV2P43BDatasetManifest,
    ) == prepared.evaluation
    assert load_json(
        output / "preparation.json",
        MobileNetV2P43BDatasetPreparationRecord,
    ) == prepared.record

    with pytest.raises(DatasetPreparationError, match="Refusing to overwrite"):
        prepare_numeric_imagefolder_manifests(
            root,
            output,
            dataset_name="imagenetv2-fixture",
            dataset_version="fixture-v1",
            provenance="Independently labelled deterministic test fixture.",
            license_description="Generated fixture for repository tests.",
            class_count=2,
            calibration_samples_per_class=1,
            evaluation_samples_per_class=2,
        )


def test_prepare_rejects_cross_split_duplicate_content(tmp_path: Path) -> None:
    root = tmp_path / "images"
    _imagefolder(root, duplicate=True)

    with pytest.raises(DatasetPreparationError, match="overlap"):
        prepare_numeric_imagefolder_manifests(
            root,
            tmp_path / "manifests",
            dataset_name="imagenetv2-fixture",
            dataset_version="fixture-v1",
            provenance="Independently labelled deterministic test fixture.",
            license_description="Generated fixture for repository tests.",
            class_count=2,
            calibration_samples_per_class=1,
            evaluation_samples_per_class=2,
        )


def test_prepare_rejects_non_numeric_inventory(tmp_path: Path) -> None:
    root = tmp_path / "images"
    _imagefolder(root)
    (root / "metadata.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match="inventory mismatch"):
        prepare_numeric_imagefolder_manifests(
            root,
            tmp_path / "manifests",
            dataset_name="imagenetv2-fixture",
            dataset_version="fixture-v1",
            provenance="Independently labelled deterministic test fixture.",
            license_description="Generated fixture for repository tests.",
            class_count=2,
            calibration_samples_per_class=1,
            evaluation_samples_per_class=2,
        )
