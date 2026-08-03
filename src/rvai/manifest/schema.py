"""Pydantic schema for RVAI model manifests."""

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveInt,
    field_validator,
)


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unsupported fields."""

    model_config = ConfigDict(extra="forbid")


class ResourceRequirements(StrictModel):
    """Coarse resource hints used while building a run plan."""

    min_memory_mb: PositiveInt
    recommended_threads: PositiveInt | Literal["auto"] = "auto"


class RiscVRequirements(StrictModel):
    """RISC-V ISA requirements and optimization preferences."""

    require_rv64: bool
    prefer_rvv: bool


class ArtifactSpec(StrictModel):
    """Verified remote model artifact declared by a Manifest."""

    filename: str = Field(min_length=1)
    url: HttpUrl
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    size_bytes: PositiveInt | None = None
    media_type: str | None = None
    license: str | None = None

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        if (
            filename in {".", ".."}
            or "\x00" in filename
            or "/" in filename
            or "\\" in filename
            or Path(filename).name != filename
        ):
            raise ValueError("artifact filename must be a plain file name")
        return filename

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, url: HttpUrl) -> HttpUrl:
        if url.scheme not in {"http", "https"}:
            raise ValueError("artifact URL must use HTTP or HTTPS")
        return url

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, sha256: str) -> str:
        return sha256.lower()


class ModelManifest(StrictModel):
    """Validated model metadata loaded from a YAML file."""

    name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    display_name: str = Field(min_length=1)
    task: Literal["chat", "image_classification", "benchmark"]
    format: Literal["gguf", "onnx", "builtin"]
    quantization: Literal["int4", "int8", "fp32"]
    runtime: Literal["llama_cpp", "onnxruntime", "builtin"]
    resources: ResourceRequirements
    riscv: RiscVRequirements
    artifact: ArtifactSpec | None = None
