"""Resolve only complete, verified artifact cache entries."""

from __future__ import annotations

from rvai.artifacts.cache import ArtifactCache
from rvai.artifacts.downloader import verify_file
from rvai.artifacts.errors import (
    ArtifactIntegrityError,
    ArtifactNotCachedError,
    ArtifactNotDeclaredError,
)
from rvai.artifacts.schema import ResolvedArtifact
from rvai.manifest import ModelManifest
from rvai.results import digest_manifest


class ArtifactResolver:
    """Return a verified local path for one artifact-bearing Manifest."""

    def __init__(self, cache: ArtifactCache | None = None) -> None:
        self.cache = cache or ArtifactCache()

    def resolve(self, manifest: ModelManifest) -> ResolvedArtifact:
        spec = manifest.artifact
        if spec is None:
            raise ArtifactNotDeclaredError(
                f"Model {manifest.name} does not declare an artifact"
            )

        path = self.cache.artifact_path(manifest.name, spec)
        if not path.is_file():
            raise ArtifactNotCachedError(
                f"Artifact for model {manifest.name} is not cached"
            )
        actual_sha256, actual_size = verify_file(path, spec)

        metadata = self.cache.load_metadata(manifest.name)
        if metadata is not None:
            expected = {
                "model": manifest.name,
                "filename": spec.filename,
                "source_url": str(spec.url),
                "sha256": spec.sha256,
                "size_bytes": actual_size,
                "manifest_digest": digest_manifest(manifest),
            }
            actual = {
                "model": metadata.model,
                "filename": metadata.filename,
                "source_url": metadata.source_url,
                "sha256": metadata.sha256,
                "size_bytes": metadata.size_bytes,
                "manifest_digest": metadata.manifest_digest,
            }
            if actual != expected:
                raise ArtifactIntegrityError(
                    f"Cached artifact metadata does not match model {manifest.name}"
                )

        return ResolvedArtifact(
            model=manifest.name,
            path=path,
            sha256=actual_sha256,
            size_bytes=actual_size,
        )
