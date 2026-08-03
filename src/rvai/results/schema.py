"""Validated persistent records for completed RVAI benchmark runs."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rvai.adapters import BenchmarkResult
from rvai.hardware import HardwareProfile


class StrictModel(BaseModel):
    """Base model that rejects unsupported record fields."""

    model_config = ConfigDict(extra="forbid")


class RunRecord(StrictModel):
    """Reproducible envelope around one validated benchmark result."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    created_at: AwareDatetime
    command: list[str] = Field(
        min_length=1,
        description="Reproducible command with secrets and credentials excluded.",
    )
    model: str = Field(min_length=1)
    target: str = Field(min_length=1)
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    hardware_profile: HardwareProfile | None = None
    result: BenchmarkResult
