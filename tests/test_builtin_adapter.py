import json
import subprocess
from pathlib import Path

import pytest

from rvai.adapters import AdapterError, BuiltinAdapter
from rvai.registry import ModelRegistry


MODELS_DIR = Path(__file__).parents[1] / "models"


def manifest():
    return ModelRegistry(MODELS_DIR).get("builtin-gemm-int8")


def payload() -> dict:
    return {
        "workload": "builtin-gemm-int8",
        "status": "success",
        "backend": "scalar",
        "execution": {
            "target_architecture": "x86_64",
            "execution_environment": "native",
            "host_architecture": "x86_64",
            "performance_representative": True,
        },
        "matrix": {"m": 256, "n": 256, "k": 256},
        "iterations": 20,
        "correctness_verified": True,
        "latency_ms": {"mean": 12.5, "p95": 13.0},
        "throughput_gops": 2.68,
        "memory_bytes": {"inputs": 131072, "output": 262144, "total": 393216},
    }


def test_build_command_uses_stable_defaults(tmp_path: Path) -> None:
    executable = tmp_path / "rvai-bench"
    executable.touch(mode=0o755)

    command = BuiltinAdapter(executable=executable).build_command(manifest())

    assert command == [
        str(executable),
        "gemm-int8",
        "--m",
        "256",
        "--n",
        "256",
        "--k",
        "256",
        "--iterations",
        "20",
        "--backend",
        "scalar",
    ]


def test_build_command_accepts_ci_workload_overrides(tmp_path: Path) -> None:
    executable = tmp_path / "rvai-bench"
    executable.touch(mode=0o755)
    adapter = BuiltinAdapter(
        executable=executable,
        environ={
            "RVAI_GEMM_M": "32",
            "RVAI_GEMM_N": "32",
            "RVAI_GEMM_K": "32",
            "RVAI_GEMM_ITERATIONS": "2",
        },
    )

    command = adapter.build_command(manifest())

    assert command[command.index("--m") + 1] == "32"
    assert command[command.index("--n") + 1] == "32"
    assert command[command.index("--k") + 1] == "32"
    assert command[command.index("--iterations") + 1] == "2"


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_build_command_rejects_invalid_workload_override(
    tmp_path: Path,
    value: str,
) -> None:
    executable = tmp_path / "rvai-bench"
    executable.touch(mode=0o755)
    adapter = BuiltinAdapter(
        executable=executable,
        environ={"RVAI_GEMM_M": value},
    )

    with pytest.raises(
        AdapterError,
        match="RVAI_GEMM_M must be a positive integer",
    ):
        adapter.build_command(manifest())


def test_execute_validates_native_json(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "rvai-bench"
    executable.touch(mode=0o755)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(payload()), stderr=""
        ),
    )

    result = BuiltinAdapter(executable=executable).execute(manifest())

    assert result.correctness_verified is True
    assert result.matrix.m == 256
    assert result.memory_bytes.total == 393216
    assert result.execution.execution_environment == "native"


def test_execute_converts_native_failure_to_adapter_error(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "rvai-bench"
    executable.touch(mode=0o755)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 2, stdout="", stderr="bad dimensions"
        ),
    )

    with pytest.raises(AdapterError, match="bad dimensions"):
        BuiltinAdapter(executable=executable).execute(manifest())


def test_execute_rejects_invalid_json(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "rvai-bench"
    executable.touch(mode=0o755)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="not-json", stderr=""
        ),
    )

    with pytest.raises(AdapterError, match="invalid JSON"):
        BuiltinAdapter(executable=executable).execute(manifest())
