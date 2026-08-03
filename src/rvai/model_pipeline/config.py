"""Load and cross-check committed MobileNetV2 P4.3B configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rvai.model_pipeline.errors import PipelineConfigError, PipelineIOError, PipelinePathError
from rvai.model_pipeline.io import load_yaml
from rvai.model_pipeline.schema import (
    MobileNetV2P43BConfiguration,
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BPipelineConfig,
    MobileNetV2P43BSourceModelConfig,
    MobileNetV2P43BSourceModelIdentity,
)


FROZEN_MOBILENET_V2_FP32_IDENTITY = MobileNetV2P43BSourceModelIdentity(
    name="mobilenetv2-12",
    format="onnx",
    precision="fp32",
    filename="mobilenetv2-12.onnx",
    size_bytes=13_964_571,
    sha256="c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
)


def load_pipeline_config(path: Path | str) -> MobileNetV2P43BPipelineConfig:
    """Load the strict frozen pipeline document."""

    try:
        return load_yaml(path, MobileNetV2P43BPipelineConfig)
    except PipelineIOError as exc:
        raise PipelineConfigError(str(exc)) from exc


def load_source_model_config(path: Path | str) -> MobileNetV2P43BSourceModelConfig:
    """Load a source identity and enforce the approved production identity."""

    try:
        source = load_yaml(path, MobileNetV2P43BSourceModelConfig)
    except PipelineIOError as exc:
        raise PipelineConfigError(str(exc)) from exc
    if source.model != FROZEN_MOBILENET_V2_FP32_IDENTITY:
        mismatches = [
            field
            for field in type(FROZEN_MOBILENET_V2_FP32_IDENTITY).model_fields
            if getattr(source.model, field)
            != getattr(FROZEN_MOBILENET_V2_FP32_IDENTITY, field)
        ]
        raise PipelineConfigError(
            "Source FP32 identity does not match the frozen MobileNetV2 identity: "
            + ", ".join(mismatches)
        )
    return source


def load_dataset_manifest(
    path: Path | str,
    *,
    expected_purpose: Literal["calibration", "evaluation"] | None = None,
) -> MobileNetV2P43BDatasetManifest:
    """Load a strict example or externally supplied dataset declaration."""

    try:
        manifest = load_yaml(path, MobileNetV2P43BDatasetManifest)
    except PipelineIOError as exc:
        raise PipelineConfigError(str(exc)) from exc
    if expected_purpose is not None and manifest.dataset.purpose != expected_purpose:
        raise PipelineConfigError(
            f"Dataset purpose is {manifest.dataset.purpose!r}, "
            f"expected {expected_purpose!r}"
        )
    return manifest


def _resolve_inside(configuration_dir: Path, relative_path: str) -> Path:
    candidate = configuration_dir / relative_path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise PipelinePathError(f"Cannot resolve configuration reference: {detail}") from exc
    if not resolved.is_relative_to(configuration_dir):
        raise PipelinePathError(
            f"Configuration reference escapes its directory: {relative_path}"
        )
    return resolved


def load_mobilenet_v2_configuration(
    configuration_dir: Path | str,
) -> MobileNetV2P43BConfiguration:
    """Load the committed pipeline and its symlink-safe source reference."""

    requested_dir = Path(configuration_dir)
    try:
        root = requested_dir.resolve(strict=True)
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise PipelinePathError(f"Cannot resolve configuration directory: {detail}") from exc
    if not root.is_dir():
        raise PipelinePathError(f"Configuration path is not a directory: {requested_dir}")

    pipeline_path = _resolve_inside(root, "pipeline.yaml")
    pipeline = load_pipeline_config(pipeline_path)
    source_path = _resolve_inside(root, pipeline.pipeline.source_model)
    source = load_source_model_config(source_path)
    return MobileNetV2P43BConfiguration(pipeline=pipeline, source=source)
