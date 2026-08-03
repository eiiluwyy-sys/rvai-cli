import hashlib

import pytest

from rvai.artifacts import (
    ArtifactCache,
    ArtifactIntegrityError,
    ArtifactNotCachedError,
    ArtifactNotDeclaredError,
    ArtifactResolver,
    CachedArtifactMetadata,
)
from rvai.manifest import ModelManifest
from rvai.results import digest_manifest


def valid_manifest() -> dict[str, object]:
    return {
        "name": "demo-model",
        "display_name": "Demo Model",
        "task": "benchmark",
        "format": "builtin",
        "quantization": "int8",
        "runtime": "builtin",
        "resources": {"min_memory_mb": 128, "recommended_threads": "auto"},
        "riscv": {"require_rv64": False, "prefer_rvv": False},
    }


def manifest_with_artifact(payload: bytes) -> ModelManifest:
    data = valid_manifest()
    data["name"] = "demo-model"
    data["artifact"] = {
        "filename": "model.bin",
        "url": "https://example.com/model.bin",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    return ModelManifest.model_validate(data)


def test_resolver_rejects_manifest_without_artifact(tmp_path) -> None:
    resolver = ArtifactResolver(ArtifactCache(root=tmp_path))
    manifest = ModelManifest.model_validate(valid_manifest())

    with pytest.raises(ArtifactNotDeclaredError, match="does not declare"):
        resolver.resolve(manifest)


def test_resolver_rejects_missing_artifact(tmp_path) -> None:
    resolver = ArtifactResolver(ArtifactCache(root=tmp_path))

    with pytest.raises(ArtifactNotCachedError, match="not cached"):
        resolver.resolve(manifest_with_artifact(b"expected"))


def test_resolver_returns_verified_file_without_metadata(tmp_path) -> None:
    payload = b"verified artifact"
    manifest = manifest_with_artifact(payload)
    cache = ArtifactCache(root=tmp_path)
    path = cache.artifact_path(manifest.name, manifest.artifact)
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    resolved = ArtifactResolver(cache).resolve(manifest)

    assert resolved.path == path
    assert resolved.sha256 == hashlib.sha256(payload).hexdigest()
    assert resolved.size_bytes == len(payload)
    assert resolved.verified is True


def test_resolver_rejects_corrupted_cached_bytes(tmp_path) -> None:
    manifest = manifest_with_artifact(b"expected")
    cache = ArtifactCache(root=tmp_path)
    path = cache.artifact_path(manifest.name, manifest.artifact)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupt")

    with pytest.raises(ArtifactIntegrityError):
        ArtifactResolver(cache).resolve(manifest)


def test_resolver_rejects_metadata_from_different_manifest(tmp_path) -> None:
    payload = b"verified artifact"
    manifest = manifest_with_artifact(payload)
    cache = ArtifactCache(root=tmp_path)
    path = cache.artifact_path(manifest.name, manifest.artifact)
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    cache.write_metadata(
        CachedArtifactMetadata(
            model=manifest.name,
            filename=manifest.artifact.filename,
            source_url=str(manifest.artifact.url),
            sha256=manifest.artifact.sha256,
            size_bytes=len(payload),
            downloaded_at="2026-08-03T10:30:00Z",
            manifest_digest=f"sha256:{'0' * 64}",
        )
    )

    with pytest.raises(ArtifactIntegrityError, match="metadata does not match"):
        ArtifactResolver(cache).resolve(manifest)


def test_resolver_accepts_matching_metadata(tmp_path) -> None:
    payload = b"verified artifact"
    manifest = manifest_with_artifact(payload)
    cache = ArtifactCache(root=tmp_path)
    path = cache.artifact_path(manifest.name, manifest.artifact)
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    cache.write_metadata(
        CachedArtifactMetadata(
            model=manifest.name,
            filename=manifest.artifact.filename,
            source_url=str(manifest.artifact.url),
            sha256=manifest.artifact.sha256,
            size_bytes=len(payload),
            downloaded_at="2026-08-03T10:30:00Z",
            manifest_digest=digest_manifest(manifest),
        )
    )

    assert ArtifactResolver(cache).resolve(manifest).path == path
