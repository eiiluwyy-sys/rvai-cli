"""Pydantic schema for RVAI model manifests."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


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


class ModelManifest(StrictModel):
    """Validated model metadata loaded from a YAML file."""

    name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    display_name: str = Field(min_length=1)
    task: Literal["chat", "image_classification", "benchmark"]
    format: Literal["gguf", "onnx", "builtin"]
    quantization: Literal["int4", "int8"]
    runtime: Literal["llama_cpp", "onnxruntime", "builtin"]
    resources: ResourceRequirements
    riscv: RiscVRequirements
