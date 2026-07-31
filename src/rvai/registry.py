"""Discovery and validation of model Manifest files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rvai.manifest import ModelManifest


class RegistryError(RuntimeError):
    """Base error for model registry failures."""


class ModelNotFoundError(RegistryError):
    """Raised when a requested model is not registered."""


def default_models_dir() -> Path:
    """Resolve the model directory for local and editable installations."""

    configured = os.getenv("RVAI_MODELS_DIR")
    if configured:
        return Path(configured).expanduser()

    current_models = Path.cwd() / "models"
    if current_models.is_dir():
        return current_models

    return Path(__file__).resolve().parents[2] / "models"


class ModelRegistry:
    """Load a directory of YAML model manifests."""

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self.models_dir = (
            Path(models_dir) if models_dir is not None else default_models_dir()
        )

    def load_all(self) -> dict[str, ModelManifest]:
        if not self.models_dir.is_dir():
            raise RegistryError(f"Model directory does not exist: {self.models_dir}")

        manifests: dict[str, ModelManifest] = {}
        paths = sorted(
            (*self.models_dir.glob("*.yaml"), *self.models_dir.glob("*.yml"))
        )
        for path in paths:
            manifest = self._load_file(path)
            if manifest.name in manifests:
                raise RegistryError(f"Duplicate model name: {manifest.name}")
            manifests[manifest.name] = manifest
        return manifests

    def list(self) -> list[ModelManifest]:
        """Return registered models ordered by name."""

        manifests = self.load_all()
        return [manifests[name] for name in sorted(manifests)]

    def get(self, name: str) -> ModelManifest:
        """Return one model or raise a user-facing registry error."""

        manifests = self.load_all()
        try:
            return manifests[name]
        except KeyError as exc:
            available = ", ".join(sorted(manifests)) or "(none)"
            raise ModelNotFoundError(
                f"Unknown model '{name}'. Available models: {available}"
            ) from exc

    @staticmethod
    def _load_file(path: Path) -> ModelManifest:
        try:
            data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RegistryError(f"Cannot read Manifest {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise RegistryError(f"Manifest must be a YAML mapping: {path}")

        try:
            return ModelManifest.model_validate(data)
        except ValidationError as exc:
            raise RegistryError(f"Invalid Manifest {path}: {exc}") from exc
