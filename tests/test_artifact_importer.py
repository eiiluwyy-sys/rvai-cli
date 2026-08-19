import hashlib

import pytest

from rvai.artifacts import (
    ArtifactCache,
    ArtifactImporter,
    ArtifactIntegrityError,
    ArtifactResolver,
)
from rvai.manifest import ModelManifest
from rvai.results import digest_manifest


def manifest(payload: bytes) -> ModelManifest:
    return ModelManifest.model_validate(
        {
            "name": "local-model",
            "display_name": "Local Model",
            "task": "image_classification",
            "format": "onnx",
            "quantization": "int8",
            "runtime": "onnxruntime",
            "resources": {"min_memory_mb": 1, "recommended_threads": "auto"},
            "riscv": {"require_rv64": False, "prefer_rvv": True},
            "artifact": {
                "filename": "local-model.onnx",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            },
        }
    )


def test_importer_verifies_and_atomically_caches_local_file(tmp_path) -> None:
    payload = b"verified local artifact"
    model = manifest(payload)
    assert model.artifact is not None
    source = tmp_path / "source.onnx"
    source.write_bytes(payload)
    cache = ArtifactCache(root=tmp_path / "cache")
    destination = cache.artifact_path(model.name, model.artifact)

    result = ArtifactImporter(cache).import_file(
        model.name,
        model.artifact,
        source,
        destination,
        manifest_digest=digest_manifest(model),
    )

    assert result.status == "imported"
    assert result.sha256 == model.artifact.sha256
    assert destination.read_bytes() == payload
    metadata = cache.load_metadata(model.name)
    assert metadata is not None
    assert metadata.source_url is None
    assert ArtifactResolver(cache).resolve(model).path == destination


def test_importer_rejects_wrong_bytes_without_publishing_partial_file(tmp_path) -> None:
    model = manifest(b"expected")
    assert model.artifact is not None
    source = tmp_path / "source.onnx"
    source.write_bytes(b"wrong")
    cache = ArtifactCache(root=tmp_path / "cache")
    destination = cache.artifact_path(model.name, model.artifact)

    with pytest.raises(ArtifactIntegrityError):
        ArtifactImporter(cache).import_file(
            model.name,
            model.artifact,
            source,
            destination,
            manifest_digest=digest_manifest(model),
        )

    assert not destination.exists()
    assert list(destination.parent.glob("*.part")) == []


def test_importer_preserves_existing_invalid_cache_without_force(tmp_path) -> None:
    payload = b"expected"
    model = manifest(payload)
    assert model.artifact is not None
    source = tmp_path / "source.onnx"
    source.write_bytes(payload)
    cache = ArtifactCache(root=tmp_path / "cache")
    destination = cache.artifact_path(model.name, model.artifact)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"user data")

    with pytest.raises(ArtifactIntegrityError, match="--force"):
        ArtifactImporter(cache).import_file(
            model.name,
            model.artifact,
            source,
            destination,
            manifest_digest=digest_manifest(model),
        )

    assert destination.read_bytes() == b"user data"
