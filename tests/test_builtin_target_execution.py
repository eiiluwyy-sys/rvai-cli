import json
import subprocess
from pathlib import Path

import pytest

from rvai.adapters import AdapterError, BuiltinAdapter
from rvai.registry import ModelRegistry


MODELS_DIR = Path(__file__).parents[1] / "models"


class FakeTarget:
    name = "qemu-riscv64"
    executable = Path("/fake/rvai-bench")

    def build_command(self, executable: Path, arguments: list[str]) -> list[str]:
        return ["qemu-riscv64", str(executable), *arguments]


def manifest():
    return ModelRegistry(MODELS_DIR).get("builtin-gemm-int8")


def qemu_payload() -> dict:
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
        "matrix": {"m": 256, "n": 256, "k": 256},
        "iterations": 20,
        "correctness_verified": True,
        "latency_ms": {"mean": 24.0, "p95": 25.0},
        "throughput_gops": 1.4,
        "memory_bytes": {"inputs": 131072, "output": 262144, "total": 393216},
    }


def test_adapter_accepts_matching_qemu_result(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(qemu_payload()), stderr=""
        ),
    )

    result = BuiltinAdapter().execute(manifest(), target=FakeTarget())

    assert result.execution.target_architecture == "riscv64"


def test_adapter_rejects_qemu_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 2, stdout="", stderr="qemu failed"
        ),
    )

    with pytest.raises(AdapterError, match="qemu failed"):
        BuiltinAdapter().execute(manifest(), target=FakeTarget())


def test_adapter_converts_qemu_timeout(monkeypatch) -> None:
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=300)

    monkeypatch.setattr(subprocess, "run", time_out)

    with pytest.raises(AdapterError, match="qemu-riscv64 workload timed out"):
        BuiltinAdapter().execute(manifest(), target=FakeTarget())


def test_adapter_rejects_invalid_qemu_json(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="not-json", stderr=""
        ),
    )

    with pytest.raises(AdapterError, match="invalid JSON"):
        BuiltinAdapter().execute(manifest(), target=FakeTarget())


def test_adapter_rejects_wrong_qemu_execution_environment(monkeypatch) -> None:
    payload = qemu_payload()
    payload["execution"] = {
        "target_architecture": "riscv64",
        "execution_environment": "native",
        "host_architecture": "riscv64",
        "performance_representative": True,
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    with pytest.raises(AdapterError, match="does not match selected target"):
        BuiltinAdapter().execute(manifest(), target=FakeTarget())
