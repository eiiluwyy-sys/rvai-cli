"""Validated hardware profile returned by ``rvai detect``."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt


class StrictModel(BaseModel):
    """Base model that rejects unsupported profile fields."""

    model_config = ConfigDict(extra="forbid")


class PlatformInfo(StrictModel):
    os: str = Field(min_length=1)
    kernel: str = Field(min_length=1)
    architecture: str = Field(min_length=1)


class CpuInfo(StrictModel):
    logical_cores: PositiveInt


class MemoryInfo(StrictModel):
    total_mb: PositiveInt
    available_mb: NonNegativeInt


class RiscVInfo(StrictModel):
    is_riscv: bool
    xlen: Literal[32, 64] | None = None
    isa: str | None = None
    extensions: list[str] = Field(default_factory=list)
    rvv: bool | None = None


class RuntimeStatus(StrictModel):
    available: bool
    version: str | None = None
    executable: str | None = None


class RuntimeInfo(StrictModel):
    builtin: RuntimeStatus
    llama_cpp: RuntimeStatus
    onnxruntime: RuntimeStatus


class HardwareProfile(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    platform: PlatformInfo
    cpu: CpuInfo
    memory: MemoryInfo
    riscv: RiscVInfo
    runtimes: RuntimeInfo
