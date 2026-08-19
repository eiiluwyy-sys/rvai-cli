"""Stable structured result schema for local text generation."""

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


class PromptInfo(StrictModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    characters: PositiveInt
    system_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    system_characters: NonNegativeInt = 0


class GenerationParameters(StrictModel):
    max_tokens: PositiveInt
    context_size: PositiveInt
    threads: PositiveInt
    batch_size: PositiveInt
    ubatch_size: PositiveInt
    temperature: NonNegativeFloat
    seed: int


class GenerationMetrics(StrictModel):
    total_ms: NonNegativeFloat
    prompt_tokens: NonNegativeInt | None = None
    prompt_eval_ms: NonNegativeFloat | None = None
    prompt_tokens_per_second: NonNegativeFloat | None = None
    generated_tokens: NonNegativeInt | None = None
    generation_ms: NonNegativeFloat | None = None
    tokens_per_second: NonNegativeFloat | None = None


class GenerationExecution(StrictModel):
    execution_environment: Literal["native"] = "native"
    provider: Literal["CPU"] = "CPU"
    runtime_version: str = Field(min_length=1)
    executable: str = Field(min_length=1)


class GenerationResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    model: str = Field(min_length=1)
    status: Literal["success"] = "success"
    runtime: Literal["llama_cpp"] = "llama_cpp"
    prompt: PromptInfo
    parameters: GenerationParameters
    response: str = Field(min_length=1)
    metrics: GenerationMetrics
    execution: GenerationExecution
