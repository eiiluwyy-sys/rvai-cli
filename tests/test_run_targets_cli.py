import json

import pytest
from typer.testing import CliRunner

import rvai.cli as cli
from rvai.adapters import BenchmarkResult


runner = CliRunner()


class FakeTarget:
    def __init__(self, name: str) -> None:
        self.name = name


def result_for(target_name: str) -> BenchmarkResult:
    is_qemu = target_name == "qemu-riscv64"
    return BenchmarkResult.model_validate(
        {
            "workload": "builtin-gemm-int8",
            "status": "success",
            "backend": "scalar",
            "execution": {
                "target_architecture": "riscv64" if is_qemu else "x86_64",
                "execution_environment": "qemu-user" if is_qemu else "native",
                "host_architecture": "x86_64",
                "performance_representative": not is_qemu,
            },
            "matrix": {"m": 256, "n": 256, "k": 256},
            "iterations": 20,
            "correctness_verified": True,
            "latency_ms": {"mean": 10.0, "p95": 11.0},
            "throughput_gops": 2.0,
            "memory_bytes": {
                "inputs": 131072,
                "output": 262144,
                "total": 393216,
            },
        }
    )


@pytest.mark.parametrize(
    ("arguments", "expected_target"),
    [
        (["run", "builtin-gemm-int8"], "native"),
        (["run", "builtin-gemm-int8", "--target", "native"], "native"),
        (
            ["run", "builtin-gemm-int8", "--target", "qemu-riscv64"],
            "qemu-riscv64",
        ),
    ],
)
def test_run_selects_execution_target(
    arguments, expected_target, monkeypatch
) -> None:
    class FakeAdapter:
        def execute(self, manifest, target):
            assert manifest.name == "builtin-gemm-int8"
            assert target.name == expected_target
            return result_for(target.name)

    monkeypatch.setattr(
        cli,
        "create_target",
        lambda name: FakeTarget(name.value),
    )
    monkeypatch.setattr(cli, "BuiltinAdapter", FakeAdapter)

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 0
    assert json.loads(result.stdout)["execution"]["execution_environment"] == (
        "qemu-user" if expected_target == "qemu-riscv64" else "native"
    )


def test_run_rejects_invalid_target() -> None:
    result = runner.invoke(
        cli.app,
        ["run", "builtin-gemm-int8", "--target", "invalid"],
    )

    assert result.exit_code != 0
    assert "invalid" in result.output.lower()
    assert "Traceback" not in result.output


def test_qemu_dry_run_does_not_create_target(monkeypatch) -> None:
    def fail_if_called(name):
        raise AssertionError("dry-run must not inspect the target environment")

    monkeypatch.setattr(cli, "create_target", fail_if_called)

    result = runner.invoke(
        cli.app,
        [
            "run",
            "builtin-gemm-int8",
            "--target",
            "qemu-riscv64",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["target"] == "qemu-riscv64"
    assert payload["dry_run"] is True
