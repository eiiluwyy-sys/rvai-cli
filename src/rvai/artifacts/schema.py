"""Validated schemas for downloaded and resolved model artifacts."""

from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PositiveInt


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CachedArtifactMetadata(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    model: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    filename: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: PositiveInt
    downloaded_at: AwareDatetime
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    verified: Literal[True] = True


class DownloadResult(StrictModel):
    status: Literal["downloaded", "already-cached"]
    model: str = Field(min_length=1)
    path: Path
    metadata_path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: PositiveInt
    verified: Literal[True] = True


class ResolvedArtifact(StrictModel):
    model: str = Field(min_length=1)
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: PositiveInt
    verified: Literal[True] = True
