"""Trust-aware comparison of validated benchmark run records."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, NonNegativeFloat, model_validator

from rvai.adapters import BenchmarkResult
from rvai.results.schema import RunRecord, StrictModel


NON_REPRESENTATIVE_MESSAGE = (
    "Performance ratio is unavailable because at least one result was produced "
    "in a non-representative execution environment."
)


class ComparedRun(StrictModel):
    """Raw result and identity fields shown on one side of a comparison."""

    run_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    target: str = Field(min_length=1)
    result: BenchmarkResult


class PerformanceComparison(StrictModel):
    """Performance ratio plus the trust decision that controls its presence."""

    available: bool
    latency_ratio_right_over_left: NonNegativeFloat | None = None
    message: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ratio_availability(self) -> "PerformanceComparison":
        has_ratio = self.latency_ratio_right_over_left is not None
        if has_ratio != self.available:
            raise ValueError("performance ratio must be present exactly when available")
        return self


class ComparisonReport(StrictModel):
    """Stable JSON returned by ``rvai compare``."""

    schema_version: Literal["1.0"] = "1.0"
    left: ComparedRun
    right: ComparedRun
    differences: list[str] = Field(default_factory=list)
    performance: PerformanceComparison


def compare_run_records(left: RunRecord, right: RunRecord) -> ComparisonReport:
    """Compare correctness and raw metrics without inventing unsafe speedups."""

    differences = _describe_differences(left, right)
    performance = _compare_performance(left, right)
    return ComparisonReport(
        left=ComparedRun(
            run_id=left.run_id,
            model=left.model,
            target=left.target,
            result=left.result,
        ),
        right=ComparedRun(
            run_id=right.run_id,
            model=right.model,
            target=right.target,
            result=right.result,
        ),
        differences=differences,
        performance=performance,
    )


def _compare_performance(
    left: RunRecord,
    right: RunRecord,
) -> PerformanceComparison:
    left_result = left.result
    right_result = right.result
    if not (
        left_result.execution.performance_representative
        and right_result.execution.performance_representative
    ):
        return PerformanceComparison(
            available=False,
            message=NON_REPRESENTATIVE_MESSAGE,
        )

    incompatible = _performance_incompatibility(left, right)
    if incompatible is not None:
        return PerformanceComparison(available=False, message=incompatible)

    if left_result.latency_ms.mean == 0:
        return PerformanceComparison(
            available=False,
            message=(
                "Performance ratio is unavailable because the left mean latency "
                "is zero."
            ),
        )

    ratio = right_result.latency_ms.mean / left_result.latency_ms.mean
    return PerformanceComparison(
        available=True,
        latency_ratio_right_over_left=ratio,
        message=(
            "Latency ratio is right / left: values below 1.0 mean right has "
            "lower latency, 1.0 means equal latency, and values above 1.0 mean "
            "right has higher latency."
        ),
    )


def _performance_incompatibility(
    left: RunRecord,
    right: RunRecord,
) -> str | None:
    left_result = left.result
    right_result = right.result
    checks = (
        (left.schema_version != right.schema_version, "record schema versions differ"),
        (left.model != right.model, "models differ"),
        (left_result.workload != right_result.workload, "workloads differ"),
        (left_result.matrix != right_result.matrix, "matrix dimensions differ"),
        (left_result.backend != right_result.backend, "backends differ"),
        (
            left_result.iterations != right_result.iterations,
            "iteration counts differ",
        ),
        (
            not (
                left_result.correctness_verified
                and right_result.correctness_verified
            ),
            "correctness was not verified for both results",
        ),
    )
    for differs, reason in checks:
        if differs:
            return f"Performance ratio is unavailable because {reason}."
    return None


def _describe_differences(left: RunRecord, right: RunRecord) -> list[str]:
    left_result = left.result
    right_result = right.result
    candidates = (
        ("model", left.model, right.model),
        ("target", left.target, right.target),
        (
            "target_architecture",
            left_result.execution.target_architecture,
            right_result.execution.target_architecture,
        ),
        (
            "execution_environment",
            left_result.execution.execution_environment,
            right_result.execution.execution_environment,
        ),
        ("workload", left_result.workload, right_result.workload),
        ("backend", left_result.backend, right_result.backend),
        (
            "matrix",
            str(left_result.matrix.model_dump()),
            str(right_result.matrix.model_dump()),
        ),
        ("iterations", left_result.iterations, right_result.iterations),
        (
            "correctness_verified",
            left_result.correctness_verified,
            right_result.correctness_verified,
        ),
    )
    return [
        f"{field}: left={left_value!r}, right={right_value!r}"
        for field, left_value, right_value in candidates
        if left_value != right_value
    ]
