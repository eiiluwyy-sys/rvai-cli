"""Manifest-driven classification Top-K output processing."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from rvai.inference.errors import InferenceOutputError
from rvai.inference.labels import classification_label
from rvai.inference.schema import ClassificationPrediction
from rvai.manifest import ClassificationOutputSpec


def classification_top_k(
    output: Any,
    spec: ClassificationOutputSpec,
    *,
    numpy: ModuleType,
) -> list[ClassificationPrediction]:
    """Convert one batch of logits or probabilities into stable Top-K output."""

    scores = numpy.asarray(output, dtype=numpy.float64)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 1 or scores.size == 0:
        raise InferenceOutputError(
            f"Expected classification output shape [1, classes], got {scores.shape}"
        )
    if not numpy.isfinite(scores).all():
        raise InferenceOutputError("Classification output contains non-finite values")

    if spec.scores == "logits":
        shifted = scores - numpy.max(scores)
        exponentials = numpy.exp(shifted)
        scores = exponentials / numpy.sum(exponentials)

    count = min(spec.top_k, int(scores.size))
    indices = numpy.argsort(-scores, kind="stable")[:count]
    try:
        return [
            ClassificationPrediction(
                index=int(index),
                label=classification_label(spec.labels, int(index)),
                score=float(scores[index]),
            )
            for index in indices
        ]
    except OSError as exc:
        raise InferenceOutputError(
            f"Cannot load classification labels {spec.labels}: {exc}"
        ) from exc
