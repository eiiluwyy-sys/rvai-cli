"""Streaming HTTP downloader with verified atomic cache replacement."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, ContextManager

from rvai.artifacts.cache import ArtifactCache
from rvai.artifacts.errors import (
    ArtifactCacheError,
    ArtifactDownloadError,
    ArtifactIntegrityError,
)
from rvai.artifacts.schema import CachedArtifactMetadata, DownloadResult
from rvai.manifest import ArtifactSpec


CHUNK_SIZE = 1024 * 1024
UrlOpen = Callable[..., ContextManager[BinaryIO]]


def inspect_file(path: Path) -> tuple[str, int]:
    """Stream one local file and return its SHA-256 and byte count."""

    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ArtifactCacheError(f"Cannot read cached artifact {path}: {exc}") from exc
    return digest.hexdigest(), size


def verify_file(path: Path, spec: ArtifactSpec) -> tuple[str, int]:
    """Verify local bytes against the Manifest declaration."""

    actual_sha256, actual_size = inspect_file(path)
    if spec.size_bytes is not None and actual_size != spec.size_bytes:
        raise ArtifactIntegrityError(
            f"Artifact size mismatch: expected {spec.size_bytes}, got {actual_size}"
        )
    if actual_sha256 != spec.sha256:
        raise ArtifactIntegrityError(
            "Artifact failed SHA-256 verification: "
            f"expected {spec.sha256}, got {actual_sha256}"
        )
    return actual_sha256, actual_size


class ArtifactDownloader:
    """Download one declared artifact without exposing partial final files."""

    def __init__(
        self,
        cache: ArtifactCache | None = None,
        urlopen: UrlOpen | None = None,
    ) -> None:
        self.cache = cache or ArtifactCache()
        self.urlopen = urlopen or urllib.request.urlopen

    def download(
        self,
        model_name: str,
        spec: ArtifactSpec,
        destination: Path,
        *,
        manifest_digest: str,
        force: bool = False,
        timeout_seconds: float = 60.0,
    ) -> DownloadResult:
        """Download, verify, atomically replace, and record cache metadata."""

        expected_destination = self.cache.artifact_path(model_name, spec)
        if destination != expected_destination:
            raise ArtifactCacheError(
                f"Artifact destination must be {expected_destination}, not {destination}"
            )

        if destination.exists() and not force:
            try:
                actual_sha256, actual_size = verify_file(destination, spec)
            except ArtifactIntegrityError as exc:
                raise ArtifactIntegrityError(
                    "Cached artifact failed verification. Use --force to replace it."
                ) from exc
            metadata = self._ensure_metadata(
                model_name,
                spec,
                destination,
                manifest_digest,
                actual_size,
            )
            return DownloadResult(
                status="already-cached",
                model=model_name,
                path=destination,
                metadata_path=metadata,
                sha256=actual_sha256,
                size_bytes=actual_size,
            )

        temporary_path: Path | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{spec.filename}.",
                suffix=".part",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                actual_sha256, actual_size = self._stream_download(
                    spec,
                    output,
                    timeout_seconds,
                )
                if spec.size_bytes is not None and actual_size != spec.size_bytes:
                    raise ArtifactIntegrityError(
                        f"Artifact size mismatch: expected {spec.size_bytes}, "
                        f"got {actual_size}"
                    )
                if actual_sha256 != spec.sha256:
                    raise ArtifactIntegrityError(
                        "Artifact failed SHA-256 verification: "
                        f"expected {spec.sha256}, got {actual_sha256}"
                    )
                output.flush()
                os.fsync(output.fileno())

            os.replace(temporary_path, destination)
            temporary_path = None
            metadata_path = self._write_metadata(
                model_name,
                spec,
                manifest_digest,
                actual_size,
                datetime.now(timezone.utc),
            )
            return DownloadResult(
                status="downloaded",
                model=model_name,
                path=destination,
                metadata_path=metadata_path,
                sha256=actual_sha256,
                size_bytes=actual_size,
            )
        except ArtifactIntegrityError:
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ArtifactDownloadError(f"Cannot download artifact: {exc}") from exc
        except OSError as exc:
            raise ArtifactDownloadError(f"Cannot store downloaded artifact: {exc}") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _stream_download(
        self,
        spec: ArtifactSpec,
        output: BinaryIO,
        timeout_seconds: float,
    ) -> tuple[str, int]:
        request = urllib.request.Request(
            str(spec.url),
            headers={"User-Agent": "rvai-cli/0.1"},
        )
        digest = hashlib.sha256()
        size = 0
        with self.urlopen(request, timeout=timeout_seconds) as response:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    def _ensure_metadata(
        self,
        model_name: str,
        spec: ArtifactSpec,
        destination: Path,
        manifest_digest: str,
        actual_size: int,
    ) -> Path:
        existing = self.cache.load_metadata(model_name)
        if existing is not None and self._metadata_matches(
            existing,
            model_name,
            spec,
            manifest_digest,
            actual_size,
        ):
            return self.cache.metadata_path(model_name)
        try:
            downloaded_at = datetime.fromtimestamp(
                destination.stat().st_mtime,
                timezone.utc,
            )
        except OSError as exc:
            raise ArtifactCacheError(
                f"Cannot inspect cached artifact {destination}: {exc}"
            ) from exc
        return self.cache.write_metadata(
            self._metadata(
                model_name,
                spec,
                manifest_digest,
                actual_size,
                downloaded_at,
            )
        )

    def _write_metadata(
        self,
        model_name: str,
        spec: ArtifactSpec,
        manifest_digest: str,
        actual_size: int,
        downloaded_at: datetime,
    ) -> Path:
        return self.cache.write_metadata(
            self._metadata(
                model_name,
                spec,
                manifest_digest,
                actual_size,
                downloaded_at,
            )
        )

    @staticmethod
    def _metadata(
        model_name: str,
        spec: ArtifactSpec,
        manifest_digest: str,
        actual_size: int,
        downloaded_at: datetime,
    ) -> CachedArtifactMetadata:
        return CachedArtifactMetadata(
            model=model_name,
            filename=spec.filename,
            source_url=str(spec.url),
            sha256=spec.sha256,
            size_bytes=actual_size,
            downloaded_at=downloaded_at,
            manifest_digest=manifest_digest,
        )

    @staticmethod
    def _metadata_matches(
        metadata: CachedArtifactMetadata,
        model_name: str,
        spec: ArtifactSpec,
        manifest_digest: str,
        actual_size: int,
    ) -> bool:
        return (
            metadata.model == model_name
            and metadata.filename == spec.filename
            and metadata.source_url == str(spec.url)
            and metadata.sha256 == spec.sha256
            and metadata.size_bytes == actual_size
            and metadata.manifest_digest == manifest_digest
            and metadata.verified is True
        )
