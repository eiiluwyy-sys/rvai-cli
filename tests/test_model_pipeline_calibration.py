import importlib.util
from pathlib import Path
import shutil

import pytest

from rvai.model_pipeline.calibration import (
    CalibrationError,
    ManifestCalibrationDataReader,
    load_model_pipeline_dependencies,
    select_calibration_samples,
)
from rvai.model_pipeline.config import load_pipeline_config
from rvai.model_pipeline.dataset import validate_dataset
from rvai.model_pipeline.io import canonical_json_bytes
from rvai.model_pipeline.schema import MobileNetV2P43BDatasetManifest


CONFIG_DIR = Path(__file__).parents[1] / "model-pipeline" / "mobilenet-v2"
COMMITTED_IMAGE = Path(__file__).parent / "fixtures" / "onnx" / "red-image.ppm"
HAS_PIPELINE_DEPENDENCIES = all(
    importlib.util.find_spec(module) is not None
    for module in ("numpy", "onnx", "onnxruntime", "PIL")
)


def calibration_manifest(paths: list[str]) -> MobileNetV2P43BDatasetManifest:
    return MobileNetV2P43BDatasetManifest.model_validate(
        {
            "schema_version": "1.0",
            "dataset": {
                "name": "calibration-dataset",
                "version": "v1",
                "split": "calibration",
                "purpose": "calibration",
                "provenance": "Tiny committed offline fixture.",
                "license": "Test fixture only.",
            },
            "preprocessing": "mobilenet-v2-imagenet-v1",
            "sample_order": "manifest",
            "samples": [
                {"id": f"sample-{index}", "path": path}
                for index, path in enumerate(paths)
            ],
        }
    )


def validated_calibration(tmp_path: Path, count: int = 3):
    root = tmp_path / "dataset"
    paths = []
    for index in range(count):
        relative = f"class-{index}/image-{index}.ppm"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(COMMITTED_IMAGE, destination)
        paths.append(relative)
    return validate_dataset(calibration_manifest(paths), root)


def test_selection_uses_exact_manifest_order(tmp_path: Path) -> None:
    dataset = validated_calibration(tmp_path)
    selection = select_calibration_samples(dataset, 2)
    assert selection.record.sample_count == 2
    assert selection.record.sample_ids == ("sample-0", "sample-1")
    assert [sample.declaration.id for sample in selection.samples] == [
        "sample-0",
        "sample-1",
    ]
    assert (
        type(selection.record).model_validate_json(
            canonical_json_bytes(selection.record)
        )
        == selection.record
    )


def test_selection_rejects_insufficient_samples(tmp_path: Path) -> None:
    dataset = validated_calibration(tmp_path, count=1)
    with pytest.raises(CalibrationError, match="requires 2 samples"):
        select_calibration_samples(dataset, 2)


@pytest.mark.parametrize("count", [0, -1, True, 1.0])
def test_selection_requires_positive_integer_count(tmp_path: Path, count) -> None:
    dataset = validated_calibration(tmp_path, count=1)
    with pytest.raises(CalibrationError, match="positive integer"):
        select_calibration_samples(dataset, count)


def test_selection_rejects_evaluation_dataset(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    shutil.copy(COMMITTED_IMAGE, root / "image.ppm")
    declared = calibration_manifest(["image.ppm"])
    evaluation = declared.model_copy(
        update={
            "dataset": declared.dataset.model_copy(update={"purpose": "evaluation"}),
            "samples": (
                declared.samples[0].model_copy(update={"label": 0}),
            ),
        }
    )
    dataset = validate_dataset(evaluation, root)
    with pytest.raises(CalibrationError, match="calibration dataset"):
        select_calibration_samples(dataset, 1)


@pytest.mark.skipif(
    not HAS_PIPELINE_DEPENDENCIES,
    reason="model-pipeline optional dependencies are not installed",
)
def test_reader_preprocesses_and_rewinds_in_manifest_order(tmp_path: Path) -> None:
    dataset = validated_calibration(tmp_path, count=2)
    selection = select_calibration_samples(dataset, 2)
    pipeline = load_pipeline_config(CONFIG_DIR / "pipeline.yaml")
    reader = ManifestCalibrationDataReader(
        selection,
        input_name="image",
        preprocessing=pipeline.preprocessing,
    )

    first = reader.get_next()
    second = reader.get_next()
    assert first is not None and second is not None
    assert tuple(first["image"].shape) == (1, 3, 224, 224)
    assert str(first["image"].dtype) == "float32"
    assert reader.get_next() is None

    reader.rewind()
    repeated = reader.get_next()
    assert repeated is not None
    assert (first["image"] == repeated["image"]).all()


def test_dependency_loader_reports_dedicated_extra(monkeypatch) -> None:
    import rvai.model_pipeline.calibration as calibration_module

    def missing_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(calibration_module.importlib, "import_module", missing_import)
    with pytest.raises(CalibrationError, match=r"\[model-pipeline\]"):
        load_model_pipeline_dependencies()
