from pathlib import Path

import pytest
from pydantic import ValidationError

from rvai.model_pipeline.schema import (
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BPipelineConfig,
    MobileNetV2P43BSourceModelConfig,
)


CONFIG_DIR = Path(__file__).parents[1] / "model-pipeline" / "mobilenet-v2"


def source_data() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "model": {
            "name": "mobilenetv2-12",
            "format": "onnx",
            "precision": "fp32",
            "filename": "mobilenetv2-12.onnx",
            "size_bytes": 13_964_571,
            "sha256": "C0C3F76D93FA3FD6580652A45618618A220FCED18BABF65774ED169DE0432AD5",
        },
    }


def dataset_data(purpose: str = "calibration") -> dict[str, object]:
    sample: dict[str, object] = {
        "id": "sample-0001",
        "path": "split/class/image.jpg",
    }
    if purpose == "evaluation":
        sample["label"] = 0
    return {
        "schema_version": "1.0",
        "dataset": {
            "name": "dataset-example",
            "version": "v1",
            "split": "validation",
            "purpose": purpose,
            "provenance": "Reviewed external dataset.",
            "license": "Dataset license.",
        },
        "preprocessing": "mobilenet-v2-imagenet-v1",
        "sample_order": "manifest",
        "samples": [sample],
    }


def test_strict_models_are_frozen_and_reject_unknown_fields() -> None:
    data = source_data()
    data["unexpected"] = True

    with pytest.raises(ValidationError, match="extra_forbidden"):
        MobileNetV2P43BSourceModelConfig.model_validate(data)

    valid = source_data()
    source = MobileNetV2P43BSourceModelConfig.model_validate(valid)
    with pytest.raises(ValidationError, match="frozen_instance"):
        source.model.filename = "changed.onnx"


def test_strict_models_reject_coercion_and_normalize_sha256() -> None:
    source = MobileNetV2P43BSourceModelConfig.model_validate(source_data())
    assert source.model.sha256 == source.model.sha256.lower()

    data = source_data()
    data["model"]["size_bytes"] = "13964571"
    with pytest.raises(ValidationError):
        MobileNetV2P43BSourceModelConfig.model_validate(data)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute/image.jpg",
        "C:/images/image.jpg",
        "images\\image.jpg",
        ".",
        "..",
        "images/./image.jpg",
        "images/../image.jpg",
        "images//image.jpg",
        "images/image.jpg/",
        " images/image.jpg",
        "images/image.jpg ",
        "images/\x00image.jpg",
    ],
)
def test_dataset_paths_must_be_canonical_posix_relative_paths(path: str) -> None:
    data = dataset_data()
    data["samples"][0]["path"] = path

    with pytest.raises(ValidationError):
        MobileNetV2P43BDatasetManifest.model_validate(data)


def test_evaluation_requires_labels_but_calibration_allows_omission() -> None:
    calibration = MobileNetV2P43BDatasetManifest.model_validate(dataset_data())
    assert calibration.samples[0].label is None

    evaluation = dataset_data("evaluation")
    del evaluation["samples"][0]["label"]
    with pytest.raises(ValidationError, match="evaluation samples require"):
        MobileNetV2P43BDatasetManifest.model_validate(evaluation)


def test_labels_are_non_negative_and_strict() -> None:
    negative = dataset_data("evaluation")
    negative["samples"][0]["label"] = -1
    with pytest.raises(ValidationError):
        MobileNetV2P43BDatasetManifest.model_validate(negative)

    coerced = dataset_data("evaluation")
    coerced["samples"][0]["label"] = "0"
    with pytest.raises(ValidationError):
        MobileNetV2P43BDatasetManifest.model_validate(coerced)


@pytest.mark.parametrize("field", ["id", "path"])
def test_dataset_samples_reject_duplicates(field: str) -> None:
    data = dataset_data()
    second = dict(data["samples"][0])
    if field == "id":
        second["path"] = "split/class/other.jpg"
    else:
        second["id"] = "sample-0002"
    data["samples"].append(second)

    with pytest.raises(ValidationError, match="must be unique"):
        MobileNetV2P43BDatasetManifest.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", " dataset-example"),
        ("version", "v1 "),
        ("provenance", "bad\x00provenance"),
        ("license", ""),
    ],
)
def test_identifiers_and_descriptions_are_bounded_and_clean(
    field: str, value: str
) -> None:
    data = dataset_data()
    data["dataset"][field] = value
    with pytest.raises(ValidationError):
        MobileNetV2P43BDatasetManifest.model_validate(data)


def test_pipeline_rejects_non_frozen_quantization_or_acceptance_values() -> None:
    import yaml

    data = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    data["quantization"]["calibration_method"] = "entropy"
    with pytest.raises(ValidationError):
        MobileNetV2P43BPipelineConfig.model_validate(data)

    data = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    data["acceptance"]["min_top1_agreement_ratio"] = 0.90
    with pytest.raises(ValidationError):
        MobileNetV2P43BPipelineConfig.model_validate(data)
