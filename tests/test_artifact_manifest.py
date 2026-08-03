from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from rvai.manifest import ArtifactSpec, ModelManifest
from rvai.registry import ModelRegistry
from rvai.results import digest_manifest


MODELS_DIR = Path(__file__).parents[1] / "models"


def artifact_data() -> dict[str, object]:
    return {
        "filename": "model.onnx",
        "url": "https://example.com/model.onnx",
        "sha256": "A" * 64,
        "size_bytes": 1024,
        "media_type": "application/onnx",
        "license": "Apache-2.0",
    }


def manifest_data() -> dict[str, object]:
    return {
        "name": "demo-fp32-onnx",
        "display_name": "Demo FP32 ONNX",
        "task": "image_classification",
        "format": "onnx",
        "quantization": "fp32",
        "runtime": "onnxruntime",
        "resources": {
            "min_memory_mb": 128,
            "recommended_threads": "auto",
        },
        "riscv": {"require_rv64": False, "prefer_rvv": True},
        "artifact": artifact_data(),
    }


def test_existing_manifests_allow_missing_artifact() -> None:
    for model in (
        "qwen-small-int4",
        "mobilenet-int8",
        "builtin-gemm-int8",
    ):
        assert ModelRegistry(MODELS_DIR).get(model).artifact is None


def test_new_mobilenet_manifest_loads_verified_declaration() -> None:
    manifest = ModelRegistry(MODELS_DIR).get("mobilenet-v2-fp32-onnx")

    assert manifest.quantization == "fp32"
    assert manifest.artifact is not None
    assert manifest.artifact.filename == "mobilenetv2-12.onnx"
    assert manifest.artifact.size_bytes == 13964571


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "../model.onnx",
        "/models/model.onnx",
        "subdir/model.onnx",
        "a\\b.onnx",
        ".",
        "..",
        "bad\x00name",
    ],
)
def test_artifact_rejects_unsafe_filename(filename: str) -> None:
    data = artifact_data()
    data["filename"] = filename

    with pytest.raises(ValidationError):
        ArtifactSpec.model_validate(data)


@pytest.mark.parametrize("sha256", ["", "a" * 63, "a" * 65, "g" * 64])
def test_artifact_rejects_invalid_sha256(sha256: str) -> None:
    data = artifact_data()
    data["sha256"] = sha256

    with pytest.raises(ValidationError):
        ArtifactSpec.model_validate(data)


def test_artifact_normalizes_sha256_to_lowercase() -> None:
    assert ArtifactSpec.model_validate(artifact_data()).sha256 == "a" * 64


def test_artifact_rejects_zero_size() -> None:
    data = artifact_data()
    data["size_bytes"] = 0

    with pytest.raises(ValidationError):
        ArtifactSpec.model_validate(data)


def test_artifact_rejects_extra_fields() -> None:
    data = artifact_data()
    data["authorization"] = "secret"

    with pytest.raises(ValidationError):
        ArtifactSpec.model_validate(data)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/model.onnx",
        "ftp://example.com/model.onnx",
        "ssh://example.com/model.onnx",
        "data:application/octet-stream;base64,AA==",
    ],
)
def test_artifact_rejects_non_http_url(url: str) -> None:
    data = artifact_data()
    data["url"] = url

    with pytest.raises(ValidationError):
        ArtifactSpec.model_validate(data)


def test_artifact_none_remains_valid() -> None:
    data = manifest_data()
    data["artifact"] = None

    assert ModelManifest.model_validate(data).artifact is None


@pytest.mark.parametrize("field", ["filename", "url", "sha256"])
def test_manifest_digest_includes_artifact_fields(field: str) -> None:
    original = ModelManifest.model_validate(manifest_data())
    changed_data = deepcopy(manifest_data())
    changed_artifact = changed_data["artifact"]
    assert isinstance(changed_artifact, dict)
    replacements = {
        "filename": "other.onnx",
        "url": "https://example.com/other.onnx",
        "sha256": "b" * 64,
    }
    changed_artifact[field] = replacements[field]

    assert digest_manifest(original) != digest_manifest(
        ModelManifest.model_validate(changed_data)
    )
