"""Adapter for the native ``rvai-bench`` builtin workloads."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    model_validator,
)

from rvai.adapters.base import WorkloadAdapter
from rvai.manifest import ModelManifest


class AdapterError(RuntimeError):
    """Raised when a workload adapter cannot execute safely."""


class StrictResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MatrixShape(StrictResultModel):
    m: PositiveInt
    n: PositiveInt
    k: PositiveInt


class LatencyStats(StrictResultModel):
    mean: NonNegativeFloat
    p95: NonNegativeFloat


class MemoryStats(StrictResultModel):
    inputs: NonNegativeInt
    output: NonNegativeInt
    total: NonNegativeInt


class ExecutionInfo(StrictResultModel):
    target_architecture: Literal[
        "x86_64", "aarch64", "riscv64", "riscv32", "unknown"
    ]
    execution_environment: Literal["native", "qemu-user"]
    host_architecture: Literal["x86_64", "aarch64", "riscv64", "riscv32"]
    performance_representative: bool

    @model_validator(mode="after")
    def validate_performance_representative(self) -> "ExecutionInfo":
        expected = self.execution_environment == "native"
        if self.performance_representative != expected:
            raise ValueError(
                "performance_representative must be false for qemu-user "
                "and true for native execution"
            )
        return self


class BenchmarkResult(StrictResultModel):
    workload: Literal["builtin-gemm-int8"]
    status: Literal["success"]
    backend: Literal["scalar"]
    execution: ExecutionInfo
    matrix: MatrixShape
    iterations: PositiveInt
    correctness_verified: bool
    latency_ms: LatencyStats
    throughput_gops: NonNegativeFloat
    memory_bytes: MemoryStats


class BuiltinAdapter(WorkloadAdapter):
    """Invoke and validate the native scalar INT8 GEMM benchmark."""

    DEFAULT_M = 256
    DEFAULT_N = 256
    DEFAULT_K = 256
    DEFAULT_ITERATIONS = 20

    def __init__(
        self,
        executable: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._executable = Path(executable) if executable is not None else None
        self._environ = os.environ if environ is None else environ

    @property
    def name(self) -> str:
        return "builtin"

    def build_command(self, manifest: ModelManifest) -> list[str]:
        if manifest.runtime != self.name or manifest.name != "builtin-gemm-int8":
            raise AdapterError(
                f"BuiltinAdapter does not support model {manifest.name}"
            )

        executable = self._resolve_executable()
        return [
            str(executable),
            "gemm-int8",
            "--m",
            str(self.DEFAULT_M),
            "--n",
            str(self.DEFAULT_N),
            "--k",
            str(self.DEFAULT_K),
            "--iterations",
            str(self.DEFAULT_ITERATIONS),
            "--backend",
            "scalar",
        ]

    def execute(self, manifest: ModelManifest) -> BenchmarkResult:
        """Run ``rvai-bench`` and validate its JSON output."""

        command = self.build_command(manifest)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except OSError as exc:
            detail = exc.strerror or str(exc)
            raise AdapterError(f"Cannot execute rvai-bench: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdapterError("rvai-bench timed out after 300 seconds") from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise AdapterError(
                f"rvai-bench failed with exit code {completed.returncode}: "
                f"{detail or 'no error output'}"
            )

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AdapterError("rvai-bench returned invalid JSON") from exc

        try:
            result = BenchmarkResult.model_validate(payload)
        except ValidationError as exc:
            raise AdapterError(f"rvai-bench returned an invalid result: {exc}") from exc

        if not result.correctness_verified:
            raise AdapterError("rvai-bench did not verify GEMM correctness")
        return result

    def _resolve_executable(self) -> Path:
        if self._executable is not None:
            return self._require_executable(self._executable)

        configured = self._environ.get("RVAI_BENCH_BIN")
        if configured:
            return self._require_executable(Path(configured).expanduser())

        discovered = shutil.which("rvai-bench")
        if discovered:
            return Path(discovered)

        project_build = Path(__file__).resolve().parents[3] / "build" / "rvai-bench"
        if project_build.is_file() and os.access(project_build, os.X_OK):
            return project_build

        raise AdapterError(
            "rvai-bench was not found. Build it with: "
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && "
            "cmake --build build --parallel"
        )

    @staticmethod
    def _require_executable(path: Path) -> Path:
        if not path.is_file() or not os.access(path, os.X_OK):
            raise AdapterError(f"rvai-bench is not executable: {path}")
        return path
