"""Adapter for the native ``rvai-bench`` builtin workloads."""

from __future__ import annotations

import json
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
from rvai.targets import ExecutionTarget, NativeTarget, TargetError


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
    host_architecture: Literal[
        "x86_64", "aarch64", "riscv64", "riscv32", "unknown"
    ]
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
    """Build GEMM arguments and validate results for any execution target."""

    DEFAULT_M = 256
    DEFAULT_N = 256
    DEFAULT_K = 256
    DEFAULT_ITERATIONS = 20

    def __init__(
        self,
        executable: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._default_target = NativeTarget(
            executable=executable,
            environ=environ,
        )

    @property
    def name(self) -> str:
        return "builtin"

    def build_workload_arguments(self, manifest: ModelManifest) -> list[str]:
        """Build arguments that are independent of the launch environment."""

        if manifest.runtime != self.name or manifest.name != "builtin-gemm-int8":
            raise AdapterError(
                f"BuiltinAdapter does not support model {manifest.name}"
            )
        return [
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

    def build_command(
        self,
        manifest: ModelManifest,
        target: ExecutionTarget | None = None,
    ) -> list[str]:
        """Combine workload arguments with one validated execution target."""

        selected_target = target or self._default_target
        arguments = self.build_workload_arguments(manifest)
        try:
            return selected_target.build_command(
                selected_target.executable,
                arguments,
            )
        except TargetError as exc:
            raise AdapterError(str(exc)) from exc

    def execute(
        self,
        manifest: ModelManifest,
        target: ExecutionTarget | None = None,
    ) -> BenchmarkResult:
        """Execute through a target and validate its JSON result."""

        selected_target = target or self._default_target
        command = self.build_command(manifest, selected_target)
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
            raise AdapterError(f"Cannot execute workload: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdapterError(
                f"{selected_target.name} workload timed out after 300 seconds"
            ) from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise AdapterError(
                f"Workload failed with exit code {completed.returncode}: "
                f"{detail or 'no error output'}"
            )

        return self.parse_result(completed.stdout, selected_target)

    def parse_result(
        self,
        stdout: str,
        target: ExecutionTarget | None = None,
    ) -> BenchmarkResult:
        """Parse native JSON and enforce its selected-target contract."""

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise AdapterError("rvai-bench returned invalid JSON") from exc

        try:
            result = BenchmarkResult.model_validate(payload)
        except ValidationError as exc:
            raise AdapterError(f"rvai-bench returned an invalid result: {exc}") from exc

        if not result.correctness_verified:
            raise AdapterError("rvai-bench did not verify GEMM correctness")
        if target is not None:
            self._validate_target_result(target, result)
        return result

    @staticmethod
    def _validate_target_result(
        target: ExecutionTarget,
        result: BenchmarkResult,
    ) -> None:
        execution = result.execution
        if target.name == "native":
            matches = execution.execution_environment == "native"
        elif target.name == "qemu-riscv64":
            matches = (
                execution.target_architecture == "riscv64"
                and execution.execution_environment == "qemu-user"
                and execution.performance_representative is False
            )
        else:
            matches = False

        if not matches:
            raise AdapterError(
                "Result execution metadata does not match selected target "
                f"{target.name}"
            )
