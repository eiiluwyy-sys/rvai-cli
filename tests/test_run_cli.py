import json

from typer.testing import CliRunner

import rvai.cli as cli
from rvai.adapters import AdapterError, BenchmarkResult


runner = CliRunner()


def benchmark_result() -> BenchmarkResult:
    return BenchmarkResult.model_validate(
        {
            "workload": "builtin-gemm-int8",
            "status": "success",
            "backend": "scalar",
            "matrix": {"m": 256, "n": 256, "k": 256},
            "iterations": 20,
            "correctness_verified": True,
            "latency_ms": {"mean": 12.5, "p95": 13.0},
            "throughput_gops": 2.68,
            "memory_bytes": {
                "inputs": 131072,
                "output": 262144,
                "total": 393216,
            },
        }
    )


def test_run_builtin_outputs_native_result(monkeypatch) -> None:
    class FakeBuiltinAdapter:
        def execute(self, manifest):
            assert manifest.name == "builtin-gemm-int8"
            return benchmark_result()

    monkeypatch.setattr(cli, "BuiltinAdapter", FakeBuiltinAdapter)

    result = runner.invoke(cli.app, ["run", "builtin-gemm-int8"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == benchmark_result().model_dump()


def test_run_builtin_hides_adapter_traceback(monkeypatch) -> None:
    class FailingBuiltinAdapter:
        def execute(self, manifest):
            raise AdapterError("native benchmark failed")

    monkeypatch.setattr(cli, "BuiltinAdapter", FailingBuiltinAdapter)

    result = runner.invoke(cli.app, ["run", "builtin-gemm-int8"])

    assert result.exit_code == 1
    assert "Error: native benchmark failed" in result.output
    assert "Traceback" not in result.output


def test_run_external_model_requires_dry_run() -> None:
    result = runner.invoke(cli.app, ["run", "qwen-small-int4"])

    assert result.exit_code == 1
    assert "supported only for builtin workloads" in result.output
    assert "Traceback" not in result.output


def test_run_dry_run_remains_available() -> None:
    result = runner.invoke(cli.app, ["run", "qwen-small-int4", "--dry-run"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["dry_run"] is True
