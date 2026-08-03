"""Safe cache layout and atomic artifact metadata persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from rvai.artifacts.errors import ArtifactCacheError
from rvai.artifacts.schema import CachedArtifactMetadata
from rvai.manifest import ArtifactSpec


_SAFE_MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ArtifactCache:
    """Determine cache paths without performing network operations."""

    def __init__(
        self,
        root: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        environment = os.environ if environ is None else environ
        configured = environment.get("RVAI_CACHE_DIR")
        if root is not None:
            self.root = Path(root).expanduser()
        elif configured:
            self.root = Path(configured).expanduser()
        else:
            self.root = Path.home() / ".cache" / "rvai" / "models"

    @staticmethod
    def validate_model_name(model_name: str) -> str:
        if not _SAFE_MODEL_NAME.fullmatch(model_name):
            raise ArtifactCacheError(f"Unsafe artifact cache model name: {model_name!r}")
        return model_name

    def artifact_dir(self, model_name: str) -> Path:
        return self.root / self.validate_model_name(model_name)

    def artifact_path(self, model_name: str, spec: ArtifactSpec) -> Path:
        return self.artifact_dir(model_name) / spec.filename

    def metadata_path(self, model_name: str) -> Path:
        return self.artifact_dir(model_name) / "artifact.json"

    def load_metadata(self, model_name: str) -> CachedArtifactMetadata | None:
        source = self.metadata_path(model_name)
        if not source.exists():
            return None
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            return CachedArtifactMetadata.model_validate(payload)
        except json.JSONDecodeError as exc:
            raise ArtifactCacheError(f"Invalid artifact metadata JSON: {source}") from exc
        except ValidationError as exc:
            raise ArtifactCacheError(f"Invalid artifact metadata {source}: {exc}") from exc
        except OSError as exc:
            raise ArtifactCacheError(f"Cannot read artifact metadata {source}: {exc}") from exc

    def write_metadata(self, metadata: CachedArtifactMetadata) -> Path:
        destination = self.metadata_path(metadata.model)
        temporary_path: Path | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            serialized = metadata.model_dump_json(indent=2) + "\n"
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=".artifact.json.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                output.write(serialized)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
            return destination
        except OSError as exc:
            raise ArtifactCacheError(
                f"Cannot write artifact metadata {destination}: {exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
