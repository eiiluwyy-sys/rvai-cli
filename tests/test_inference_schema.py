import math

import pytest
from pydantic import ValidationError

from rvai.inference import (
    ClassificationPrediction,
    ExecutionInfo,
    InferenceResult,
    InputInfo,
)


def valid_result() -> InferenceResult:
    return InferenceResult(
        model="tiny-classifier",
        input=InputInfo(
            path="image.ppm",
            original_width=2,
            original_height=2,
            tensor_shape=(1, 3, 2, 2),
        ),
        predictions=[
            ClassificationPrediction(index=0, label="rgb:0", score=0.9)
        ],
        execution=ExecutionInfo(
            provider="CPUExecutionProvider",
            runtime_version="1.28.0",
            latency_ms=0.1,
        ),
    )


def test_inference_result_has_independent_versioned_schema() -> None:
    result = valid_result()

    assert result.schema_version == "1.0"
    assert result.status == "success"
    assert result.runtime == "onnxruntime"
    assert result.execution.execution_environment == "native"


def test_inference_result_rejects_unknown_fields() -> None:
    payload = valid_result().model_dump()
    payload["benchmark"] = True

    with pytest.raises(ValidationError):
        InferenceResult.model_validate(payload)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_prediction_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValidationError):
        ClassificationPrediction(index=0, label="rgb:0", score=score)
