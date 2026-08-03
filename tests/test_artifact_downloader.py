import hashlib
import urllib.error
from pathlib import Path

import pytest

import rvai.artifacts.downloader as downloader_module
from rvai.artifacts import (
    CHUNK_SIZE,
    ArtifactCache,
    ArtifactCacheError,
    ArtifactDownloadError,
    ArtifactDownloader,
    ArtifactIntegrityError,
)
from rvai.manifest import ArtifactSpec


MANIFEST_DIGEST = f"sha256:{'d' * 64}"


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0
        self.read_sizes: list[int] = []

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class FakeUrlOpen:
    def __init__(self, payload: bytes | None = None, error: Exception | None = None) -> None:
        self.payload = payload or b""
        self.error = error
        self.calls = 0
        self.response: FakeResponse | None = None

    def __call__(self, request: object, *, timeout: float) -> FakeResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        self.response = FakeResponse(self.payload)
        return self.response


def artifact_spec(
    payload: bytes,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> ArtifactSpec:
    return ArtifactSpec(
        filename="model.onnx",
        url="https://example.com/model.onnx",
        sha256=sha256 or hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload) if size_bytes is None else size_bytes,
    )


def downloader(tmp_path, opener: FakeUrlOpen) -> tuple[ArtifactDownloader, ArtifactCache]:
    cache = ArtifactCache(root=tmp_path / "cache")
    return ArtifactDownloader(cache=cache, urlopen=opener), cache


def assert_no_partial_files(destination: Path) -> None:
    assert not list(destination.parent.glob("*.part"))


def test_download_streams_verifies_and_writes_metadata(tmp_path) -> None:
    payload = b"x" * (CHUNK_SIZE + 17)
    opener = FakeUrlOpen(payload)
    subject, cache = downloader(tmp_path, opener)
    spec = artifact_spec(payload)
    destination = cache.artifact_path("demo-model", spec)

    result = subject.download(
        "demo-model",
        spec,
        destination,
        manifest_digest=MANIFEST_DIGEST,
    )

    assert result.status == "downloaded"
    assert result.verified is True
    assert destination.read_bytes() == payload
    assert cache.load_metadata("demo-model").manifest_digest == MANIFEST_DIGEST
    assert opener.response is not None
    assert all(size == CHUNK_SIZE for size in opener.response.read_sizes)
    assert len(opener.response.read_sizes) >= 3
    assert_no_partial_files(destination)


def test_hash_mismatch_does_not_publish_final_file(tmp_path) -> None:
    payload = b"downloaded bytes"
    opener = FakeUrlOpen(payload)
    subject, cache = downloader(tmp_path, opener)
    spec = artifact_spec(payload, sha256="0" * 64)
    destination = cache.artifact_path("demo-model", spec)

    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        subject.download(
            "demo-model", spec, destination, manifest_digest=MANIFEST_DIGEST
        )

    assert not destination.exists()
    assert not cache.metadata_path("demo-model").exists()
    assert_no_partial_files(destination)


def test_size_mismatch_does_not_publish_final_file(tmp_path) -> None:
    payload = b"downloaded bytes"
    opener = FakeUrlOpen(payload)
    subject, cache = downloader(tmp_path, opener)
    spec = artifact_spec(payload, size_bytes=len(payload) + 1)
    destination = cache.artifact_path("demo-model", spec)

    with pytest.raises(ArtifactIntegrityError, match="size mismatch"):
        subject.download(
            "demo-model", spec, destination, manifest_digest=MANIFEST_DIGEST
        )

    assert not destination.exists()
    assert_no_partial_files(destination)


def test_network_failure_is_wrapped_and_leaves_no_partial_file(tmp_path) -> None:
    opener = FakeUrlOpen(error=urllib.error.URLError("offline"))
    subject, cache = downloader(tmp_path, opener)
    spec = artifact_spec(b"expected")
    destination = cache.artifact_path("demo-model", spec)

    with pytest.raises(ArtifactDownloadError, match="Cannot download artifact"):
        subject.download(
            "demo-model", spec, destination, manifest_digest=MANIFEST_DIGEST
        )

    assert not destination.exists()
    assert_no_partial_files(destination)


def test_valid_cache_is_not_downloaded_again(tmp_path) -> None:
    payload = b"valid cache"
    opener = FakeUrlOpen(error=AssertionError("network must not be called"))
    subject, cache = downloader(tmp_path, opener)
    spec = artifact_spec(payload)
    destination = cache.artifact_path("demo-model", spec)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)

    result = subject.download(
        "demo-model", spec, destination, manifest_digest=MANIFEST_DIGEST
    )

    assert result.status == "already-cached"
    assert opener.calls == 0
    assert cache.load_metadata("demo-model") is not None


def test_invalid_cache_requires_force_and_is_preserved(tmp_path) -> None:
    valid_payload = b"expected"
    invalid_payload = b"user file"
    opener = FakeUrlOpen(valid_payload)
    subject, cache = downloader(tmp_path, opener)
    spec = artifact_spec(valid_payload)
    destination = cache.artifact_path("demo-model", spec)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(invalid_payload)

    with pytest.raises(ArtifactIntegrityError, match="--force"):
        subject.download(
            "demo-model", spec, destination, manifest_digest=MANIFEST_DIGEST
        )

    assert destination.read_bytes() == invalid_payload
    assert opener.calls == 0


def test_force_failure_preserves_existing_cache(tmp_path) -> None:
    old_payload = b"old cache"
    expected_payload = b"new cache"
    opener = FakeUrlOpen(error=urllib.error.URLError("offline"))
    subject, cache = downloader(tmp_path, opener)
    spec = artifact_spec(expected_payload)
    destination = cache.artifact_path("demo-model", spec)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(old_payload)

    with pytest.raises(ArtifactDownloadError):
        subject.download(
            "demo-model",
            spec,
            destination,
            manifest_digest=MANIFEST_DIGEST,
            force=True,
        )

    assert destination.read_bytes() == old_payload
    assert_no_partial_files(destination)


def test_force_success_atomically_replaces_existing_cache(tmp_path) -> None:
    old_payload = b"old cache"
    expected_payload = b"new cache"
    opener = FakeUrlOpen(expected_payload)
    subject, cache = downloader(tmp_path, opener)
    spec = artifact_spec(expected_payload)
    destination = cache.artifact_path("demo-model", spec)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(old_payload)

    result = subject.download(
        "demo-model",
        spec,
        destination,
        manifest_digest=MANIFEST_DIGEST,
        force=True,
    )

    assert result.status == "downloaded"
    assert destination.read_bytes() == expected_payload


def test_atomic_replace_failure_preserves_existing_cache(tmp_path, monkeypatch) -> None:
    old_payload = b"old cache"
    expected_payload = b"new cache"
    opener = FakeUrlOpen(expected_payload)
    subject, cache = downloader(tmp_path, opener)
    spec = artifact_spec(expected_payload)
    destination = cache.artifact_path("demo-model", spec)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(old_payload)

    def fail_replace(source, target) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(downloader_module.os, "replace", fail_replace)

    with pytest.raises(ArtifactDownloadError, match="Cannot store"):
        subject.download(
            "demo-model",
            spec,
            destination,
            manifest_digest=MANIFEST_DIGEST,
            force=True,
        )

    assert destination.read_bytes() == old_payload
    assert_no_partial_files(destination)


def test_downloader_rejects_destination_outside_cache_layout(tmp_path) -> None:
    payload = b"expected"
    opener = FakeUrlOpen(payload)
    subject, _ = downloader(tmp_path, opener)
    spec = artifact_spec(payload)

    with pytest.raises(ArtifactCacheError, match="destination must be"):
        subject.download(
            "demo-model",
            spec,
            tmp_path / "elsewhere.onnx",
            manifest_digest=MANIFEST_DIGEST,
        )

    assert opener.calls == 0
