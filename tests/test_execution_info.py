import pytest
from pydantic import ValidationError

from rvai.adapters import BenchmarkResult


def result_payload() -> dict:
    return {
        "workload": "builtin-gemm-int8",
        "status": "success",
        "backend": "scalar",
        "execution": {
            "target_architecture": "riscv64",
            "execution_environment": "qemu-user",
            "host_architecture": "x86_64",
            "performance_representative": False,
        },
        "matrix": {"m": 32, "n": 32, "k": 32},
        "iterations": 2,
        "correctness_verified": True,
        "latency_ms": {"mean": 1.0, "p95": 1.1},
        "throughput_gops": 0.1,
        "memory_bytes": {"inputs": 2048, "output": 4096, "total": 6144},
    }


def test_qemu_execution_metadata_is_valid() -> None:
    result = BenchmarkResult.model_validate(result_payload())

    assert result.execution.target_architecture == "riscv64"
    assert result.execution.execution_environment == "qemu-user"
    assert result.execution.performance_representative is False


def test_qemu_performance_cannot_be_marked_representative() -> None:
    payload = result_payload()
    payload["execution"]["performance_representative"] = True

    with pytest.raises(ValidationError, match="must be false for qemu-user"):
        BenchmarkResult.model_validate(payload)
