"""Stable result schema for single-image classification inference."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ClassificationPrediction(StrictModel):
    index: NonNegativeInt
    label: str = Field(min_length=1)
    score: float


class InputInfo(StrictModel):
    path: str = Field(min_length=1)
    media_type: Literal["image"] = "image"
    original_width: PositiveInt
    original_height: PositiveInt
    tensor_shape: tuple[PositiveInt, PositiveInt, PositiveInt, PositiveInt]
    dtype: Literal["float32"] = "float32"


class ExecutionInfo(StrictModel):
    execution_environment: Literal["native"] = "native"
    provider: str = Field(min_length=1)
    runtime_version: str = Field(min_length=1)
    latency_ms: NonNegativeFloat


class InferenceResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    model: str = Field(min_length=1)
    status: Literal["success"] = "success"
    runtime: Literal["onnxruntime"] = "onnxruntime"
    input: InputInfo
    predictions: list[ClassificationPrediction] = Field(min_length=1)
    execution: ExecutionInfo
