from datetime import datetime, timezone

import pytest

from rvai.adapters import BenchmarkResult
from rvai.results import RunRecord


@pytest.fixture
def benchmark_result_factory():
    def build(
        *,
        representative: bool = True,
        execution_environment: str = "native",
        target_architecture: str = "x86_64",
        m: int = 256,
        n: int = 256,
        k: int = 256,
        iterations: int = 20,
        mean_latency: float = 10.0,
        correctness_verified: bool = True,
    ) -> BenchmarkResult:
        return BenchmarkResult.model_validate(
            {
                "workload": "builtin-gemm-int8",
                "status": "success",
                "backend": "scalar",
                "execution": {
                    "target_architecture": target_architecture,
                    "execution_environment": execution_environment,
                    "host_architecture": "x86_64",
                    "performance_representative": representative,
                },
                "matrix": {"m": m, "n": n, "k": k},
                "iterations": iterations,
                "correctness_verified": correctness_verified,
                "latency_ms": {
                    "mean": mean_latency,
                    "p95": mean_latency * 1.1,
                },
                "throughput_gops": 2.0,
                "memory_bytes": {
                    "inputs": 131072,
                    "output": 262144,
                    "total": 393216,
                },
            }
        )

    return build


@pytest.fixture
def run_record_factory(benchmark_result_factory):
    def build(
        *,
        result: BenchmarkResult | None = None,
        model: str = "builtin-gemm-int8",
        target: str = "native",
        run_id: str = "run-left",
    ) -> RunRecord:
        return RunRecord(
            run_id=run_id,
            created_at=datetime(2026, 8, 3, 1, 2, 3, tzinfo=timezone.utc),
            command=["rvai", "run", model, "--target", target],
            model=model,
            target=target,
            manifest_digest="sha256:" + "a" * 64,
            hardware_profile=None,
            result=result or benchmark_result_factory(),
        )

    return build
