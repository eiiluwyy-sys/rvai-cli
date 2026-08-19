"""Strict, versioned schemas for the MobileNetV2 P4.3B pipeline."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: Final[Literal["1.0"]] = "1.0"
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


class StrictModel(BaseModel):
    """Immutable pipeline record that rejects coercion and unknown data."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
        revalidate_instances="always",
    )


def _clean_string(value: str) -> str:
    if "\x00" in value:
        raise ValueError("must not contain NUL")
    if value != value.strip():
        raise ValueError("must not contain leading or trailing whitespace")
    return value


Identifier = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    AfterValidator(_clean_string),
]
Description = Annotated[
    str,
    Field(min_length=1, max_length=2048),
    AfterValidator(_clean_string),
]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]


def _canonical_relative_path(value: str) -> str:
    _clean_string(value)
    if (
        not value
        or value in {".", ".."}
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or "//" in value
        or _DRIVE_PREFIX.match(value)
    ):
        raise ValueError("must be a canonical POSIX relative path")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("must not contain empty, current, or parent components")
    if path.as_posix() != value:
        raise ValueError("must already be in canonical POSIX form")
    return value


CanonicalRelativePath = Annotated[
    str,
    Field(min_length=1, max_length=1024),
    AfterValidator(_canonical_relative_path),
]


def _plain_filename(value: str) -> str:
    _canonical_relative_path(value)
    if len(PurePosixPath(value).parts) != 1:
        raise ValueError("must be a plain file name")
    return value


PlainFilename = Annotated[
    str,
    Field(min_length=1, max_length=255),
    AfterValidator(_plain_filename),
]


class MobileNetV2P43BSourceModelIdentity(StrictModel):
    """Frozen identity of the FP32 model consumed by P4.3B."""

    name: Literal["mobilenetv2-12"]
    format: Literal["onnx"]
    precision: Literal["fp32"]
    filename: PlainFilename
    size_bytes: PositiveInt
    sha256: Sha256Digest

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


class MobileNetV2P43BSourceModelConfig(StrictModel):
    """Versioned committed source-model identity document."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    model: MobileNetV2P43BSourceModelIdentity


class MobileNetV2P43BPipelineIdentity(StrictModel):
    """Identity and local cross-file reference for this pipeline revision."""

    name: Literal["mobilenet-v2-int8"]
    version: Identifier
    source_model: CanonicalRelativePath


class MobileNetV2P43BNormalizationConfig(StrictModel):
    """Frozen ImageNet channel normalization values."""

    scale: float = Field(gt=0)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @field_validator("mean", "std", mode="before")
    @classmethod
    def list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("std")
    @classmethod
    def positive_std(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if any(item <= 0 for item in value):
            raise ValueError("normalization standard deviations must be positive")
        return value


class MobileNetV2P43BPreprocessingConfig(StrictModel):
    """P4.3A-compatible MobileNetV2 image preprocessing contract."""

    contract: Literal["mobilenet-v2-imagenet-v1"]
    media_type: Literal["image"]
    color_space: Literal["rgb"]
    width: Literal[224]
    height: Literal[224]
    layout: Literal["nchw"]
    dtype: Literal["float32"]
    resize: Literal["bilinear"]
    resize_shorter: Literal[256]
    crop: Literal["center"]
    normalize: MobileNetV2P43BNormalizationConfig


class MobileNetV2P43BQuantizationConfig(StrictModel):
    """Frozen first-production quantization settings."""

    method: Literal["static"]
    format: Literal["qdq"]
    activation_type: Literal["quint8"]
    weight_type: Literal["qint8"]
    per_channel: Literal[True]
    calibration_method: Literal["minmax"]
    execution_provider: Literal["CPUExecutionProvider"]
    calibration_order: Literal["manifest"]


class MobileNetV2P43BQuantizationV2Config(StrictModel):
    """Explicit reviewed settings for post-baseline quantization experiments."""

    revision: Literal["v2"]
    method: Literal["static"]
    format: Literal["qdq"]
    activation_type: Literal["quint8"]
    weight_type: Literal["qint8"]
    per_channel: Literal[True]
    calibration_method: Literal["minmax", "percentile", "entropy"]
    calibration_percentile: float | None
    calibration_chunk_size: PositiveInt | None
    execution_provider: Literal["CPUExecutionProvider"]
    calibration_order: Literal["manifest"]
    op_types_to_quantize: tuple[Identifier, ...] = Field(min_length=1)
    nodes_to_exclude: tuple[Identifier, ...]
    op_types_to_exclude_output_quantization: tuple[Identifier, ...]

    @field_validator(
        "op_types_to_quantize",
        "nodes_to_exclude",
        "op_types_to_exclude_output_quantization",
        mode="before",
    )
    @classmethod
    def list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_explicit_settings(self) -> "MobileNetV2P43BQuantizationV2Config":
        if len(set(self.op_types_to_quantize)) != len(self.op_types_to_quantize):
            raise ValueError("op_types_to_quantize must not contain duplicates")
        if len(set(self.nodes_to_exclude)) != len(self.nodes_to_exclude):
            raise ValueError("nodes_to_exclude must not contain duplicates")
        if len(set(self.op_types_to_exclude_output_quantization)) != len(
            self.op_types_to_exclude_output_quantization
        ):
            raise ValueError(
                "op_types_to_exclude_output_quantization must not contain duplicates"
            )
        unknown_output_exclusions = set(
            self.op_types_to_exclude_output_quantization
        ) - set(self.op_types_to_quantize)
        if unknown_output_exclusions:
            raise ValueError(
                "output-quantization exclusions must be present in op_types_to_quantize"
            )
        if self.calibration_method == "percentile":
            if self.calibration_percentile is None or not (
                0.0 < self.calibration_percentile < 100.0
            ):
                raise ValueError(
                    "percentile calibration requires calibration_percentile in (0, 100)"
                )
        elif self.calibration_percentile is not None:
            raise ValueError(
                "calibration_percentile is only valid for percentile calibration"
            )
        if self.calibration_method in {"percentile", "entropy"}:
            if self.calibration_chunk_size is None:
                raise ValueError(
                    "histogram calibration requires calibration_chunk_size"
                )
        return self


class MobileNetV2P43BSampleCountConfig(StrictModel):
    """Pilot and production sample counts for one pipeline stage."""

    pilot_samples: PositiveInt
    production_samples: PositiveInt

    @model_validator(mode="after")
    def production_covers_pilot(self) -> "MobileNetV2P43BSampleCountConfig":
        if self.production_samples < self.pilot_samples:
            raise ValueError("production_samples must be at least pilot_samples")
        return self


class MobileNetV2P43BAcceptanceConfig(StrictModel):
    """Frozen engineering acceptance gates for the first production run."""

    # Python's typing specification excludes float Literal values, while
    # Pydantic intentionally uses them here as exact runtime configuration
    # locks for the frozen first production run.
    max_top1_drop_percentage_points: Literal[1.0]  # type: ignore[valid-type]
    max_top5_drop_percentage_points: Literal[1.0]  # type: ignore[valid-type]
    min_model_size_reduction_ratio: Literal[0.50]  # type: ignore[valid-type]
    min_top1_agreement_ratio: Literal[0.95]  # type: ignore[valid-type]
    require_zero_inference_failures: Literal[True]
    require_finite_outputs: Literal[True]


class MobileNetV2P43BAcceptanceV3Config(StrictModel):
    """Reviewed v3 gates for a non-safety-critical edge classifier."""

    revision: Literal["v3"]
    max_top1_drop_percentage_points: Literal[1.0]  # type: ignore[valid-type]
    max_top5_drop_percentage_points: Literal[1.0]  # type: ignore[valid-type]
    min_model_size_reduction_ratio: Literal[0.50]  # type: ignore[valid-type]
    min_top1_agreement_ratio: Literal[0.90]  # type: ignore[valid-type]
    max_correct_to_wrong_regression_ratio: Literal[0.03]  # type: ignore[valid-type]
    min_mean_top5_overlap_ratio: Literal[0.85]  # type: ignore[valid-type]
    require_zero_inference_failures: Literal[True]
    require_finite_outputs: Literal[True]


class MobileNetV2P43BPipelineConfig(StrictModel):
    """Complete deterministic configuration for MobileNetV2 P4.3B."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    pipeline: MobileNetV2P43BPipelineIdentity
    preprocessing: MobileNetV2P43BPreprocessingConfig
    quantization: (
        MobileNetV2P43BQuantizationConfig | MobileNetV2P43BQuantizationV2Config
    )
    calibration: MobileNetV2P43BSampleCountConfig
    evaluation: MobileNetV2P43BSampleCountConfig
    acceptance: MobileNetV2P43BAcceptanceConfig | MobileNetV2P43BAcceptanceV3Config

    @model_validator(mode="after")
    def calibration_chunks_cover_configured_samples(
        self,
    ) -> "MobileNetV2P43BPipelineConfig":
        quantization = self.quantization
        if isinstance(quantization, MobileNetV2P43BQuantizationV2Config):
            chunk_size = quantization.calibration_chunk_size
            if chunk_size is not None and (
                self.calibration.pilot_samples % chunk_size != 0
                or self.calibration.production_samples % chunk_size != 0
            ):
                raise ValueError(
                    "calibration_chunk_size must divide pilot and production samples"
                )
        return self


class MobileNetV2P43BDatasetIdentity(StrictModel):
    """Identity, purpose, and provenance for one external dataset split."""

    name: Identifier
    version: Identifier
    split: Identifier
    purpose: Literal["calibration", "evaluation"]
    provenance: Description
    license: Description


class MobileNetV2P43BDatasetSample(StrictModel):
    """One deterministically ordered external sample declaration."""

    id: Identifier
    path: CanonicalRelativePath
    label: NonNegativeInt | None = None
    sha256: Sha256Digest | None = None

    @field_validator("sha256")
    @classmethod
    def normalize_optional_sha256(cls, value: str | None) -> str | None:
        return value.lower() if value is not None else None


class MobileNetV2P43BDatasetManifest(StrictModel):
    """Versioned ordered calibration or labelled evaluation declaration."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    dataset: MobileNetV2P43BDatasetIdentity
    preprocessing: Literal["mobilenet-v2-imagenet-v1"]
    sample_order: Literal["manifest"]
    samples: tuple[MobileNetV2P43BDatasetSample, ...] = Field(min_length=1)

    @field_validator("samples", mode="before")
    @classmethod
    def sample_list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_samples(self) -> "MobileNetV2P43BDatasetManifest":
        identifiers = [sample.id for sample in self.samples]
        paths = [sample.path for sample in self.samples]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("dataset sample identifiers must be unique")
        if len(set(paths)) != len(paths):
            raise ValueError("dataset sample paths must be unique")
        if self.dataset.purpose == "evaluation":
            missing = [sample.id for sample in self.samples if sample.label is None]
            if missing:
                raise ValueError("evaluation samples require non-negative labels")
        return self


class MobileNetV2P43BConfiguration(StrictModel):
    """Validated committed configuration without resolved host paths."""

    pipeline: MobileNetV2P43BPipelineConfig
    source: MobileNetV2P43BSourceModelConfig
