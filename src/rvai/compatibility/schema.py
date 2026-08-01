"""Stable compatibility report returned by ``rvai check``."""

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unsupported report fields."""

    model_config = ConfigDict(extra="forbid")


class CompatibilityIssue(StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class CompatibilityReport(StrictModel):
    model: str = Field(min_length=1)
    compatible: bool
    ready: bool
    blocking_reasons: list[CompatibilityIssue] = Field(default_factory=list)
    warnings: list[CompatibilityIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
