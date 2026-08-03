import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import rvai.artifacts.cache as cache_module
from rvai.artifacts import ArtifactCache, ArtifactCacheError, CachedArtifactMetadata
from rvai.manifest import ArtifactSpec


SHA256 = "a" * 64
MANIFEST_DIGEST = f"sha256:{'b' * 64}"


def artifact_spec() -> ArtifactSpec:
    return ArtifactSpec(
        filename="model.onnx",
        url="https://example.com/model.onnx",
        sha256=SHA256,
        size_bytes=10,
    )


def metadata() -> CachedArtifactMetadata:
    return CachedArtifactMetadata(
        model="demo-model",
        filename="model.onnx",
        source_url="https://example.com/model.onnx",
        sha256=SHA256,
        size_bytes=10,
        downloaded_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
        manifest_digest=MANIFEST_DIGEST,
    )


def test_cache_uses_environment_override(tmp_path) -> None:
    root = tmp_path / "external-cache"

    cache = ArtifactCache(environ={"RVAI_CACHE_DIR": str(root)})

    assert cache.root == root
    assert cache.artifact_path("demo-model", artifact_spec()) == (
        root / "demo-model" / "model.onnx"
    )
    assert cache.metadata_path("demo-model") == root / "demo-model" / "artifact.json"


def test_cache_default_is_outside_the_repository() -> None:
    cache = ArtifactCache(environ={})

    assert cache.root == Path.home() / ".cache" / "rvai" / "models"


@pytest.mark.parametrize(
    "model_name",
    ["", "../demo", "demo/model", "Demo", ".hidden", "demo model"],
)
def test_cache_rejects_unsafe_model_names(tmp_path, model_name: str) -> None:
    cache = ArtifactCache(root=tmp_path)

    with pytest.raises(ArtifactCacheError, match="Unsafe"):
        cache.artifact_dir(model_name)


def test_metadata_is_written_and_loaded_atomically(tmp_path) -> None:
    cache = ArtifactCache(root=tmp_path)

    destination = cache.write_metadata(metadata())

    assert destination == tmp_path / "demo-model" / "artifact.json"
    assert cache.load_metadata("demo-model") == metadata()
    assert not list(destination.parent.glob(".artifact.json.*.tmp"))


def test_metadata_replace_failure_preserves_existing_file(tmp_path, monkeypatch) -> None:
    cache = ArtifactCache(root=tmp_path)
    destination = cache.write_metadata(metadata())
    original = destination.read_bytes()
    replacement = metadata().model_copy(update={"manifest_digest": f"sha256:{'c' * 64}"})

    def fail_replace(source, target) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(cache_module.os, "replace", fail_replace)

    with pytest.raises(ArtifactCacheError, match="Cannot write artifact metadata"):
        cache.write_metadata(replacement)

    assert destination.read_bytes() == original
    assert not list(destination.parent.glob(".artifact.json.*.tmp"))


def test_invalid_metadata_json_is_wrapped_as_cache_error(tmp_path) -> None:
    cache = ArtifactCache(root=tmp_path)
    destination = cache.metadata_path("demo-model")
    destination.parent.mkdir(parents=True)
    destination.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ArtifactCacheError, match="Invalid artifact metadata JSON"):
        cache.load_metadata("demo-model")


def test_metadata_rejects_unsupported_schema_version(tmp_path) -> None:
    cache = ArtifactCache(root=tmp_path)
    destination = cache.metadata_path("demo-model")
    destination.parent.mkdir(parents=True)
    payload = metadata().model_dump(mode="json")
    payload["schema_version"] = "2.0"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactCacheError, match="Invalid artifact metadata"):
        cache.load_metadata("demo-model")
