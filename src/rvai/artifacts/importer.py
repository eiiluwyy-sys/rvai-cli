"""Verified atomic import of pre-provisioned model artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rvai.artifacts.cache import ArtifactCache
from rvai.artifacts.downloader import CHUNK_SIZE, verify_file
from rvai.artifacts.errors import (
    ArtifactCacheError,
    ArtifactImportError,
    ArtifactIntegrityError,
)
from rvai.artifacts.schema import CachedArtifactMetadata, ImportResult
from rvai.manifest import ArtifactSpec


class ArtifactImporter:
    """Copy one local file into the verified cache without network access."""

    def __init__(self, cache: ArtifactCache | None = None) -> None:
        self.cache = cache or ArtifactCache()

    def import_file(
        self,
        model_name: str,
        spec: ArtifactSpec,
        source: Path,
        destination: Path,
        *,
        manifest_digest: str,
        force: bool = False,
    ) -> ImportResult:
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
            self._write_metadata(model_name, spec, manifest_digest, actual_size)
            return ImportResult(
                status="already-cached",
                model=model_name,
                path=destination,
                sha256=actual_sha256,
                size_bytes=actual_size,
            )
        if not source.is_file():
            raise ArtifactImportError(f"Local artifact is not a readable file: {source}")

        temporary_path: Path | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as input_file, tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{spec.filename}.",
                suffix=".part",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = input_file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                actual_sha256 = digest.hexdigest()
                if spec.size_bytes is not None and size != spec.size_bytes:
                    raise ArtifactIntegrityError(
                        f"Artifact size mismatch: expected {spec.size_bytes}, got {size}"
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
            self._write_metadata(model_name, spec, manifest_digest, size)
            return ImportResult(
                status="imported",
                model=model_name,
                path=destination,
                sha256=actual_sha256,
                size_bytes=size,
            )
        except ArtifactIntegrityError:
            raise
        except OSError as exc:
            raise ArtifactImportError(f"Cannot import local artifact: {exc}") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _write_metadata(
        self,
        model_name: str,
        spec: ArtifactSpec,
        manifest_digest: str,
        size_bytes: int,
    ) -> Path:
        return self.cache.write_metadata(
            CachedArtifactMetadata(
                model=model_name,
                filename=spec.filename,
                source_url=str(spec.url) if spec.url is not None else None,
                sha256=spec.sha256,
                size_bytes=size_bytes,
                downloaded_at=datetime.now(timezone.utc),
                manifest_digest=manifest_digest,
            )
        )
