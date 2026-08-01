"""Compatibility rules between model Manifests and Hardware Profiles."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from pathlib import Path

from pydantic import ValidationError

from rvai.compatibility.schema import CompatibilityIssue, CompatibilityReport
from rvai.hardware.schema import HardwareProfile
from rvai.manifest import ModelManifest


class CompatibilityError(RuntimeError):
    """Raised when compatibility input cannot be loaded or validated."""


def load_hardware_profile(path: Path | str) -> HardwareProfile:
    """Load a validated Hardware Profile from a JSON file."""

    profile_path = Path(path)
    try:
        raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise CompatibilityError(
            f"Cannot read Hardware Profile {profile_path}: {detail}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CompatibilityError(
            f"Invalid Hardware Profile JSON in {profile_path}: {exc.msg}"
        ) from exc

    try:
        return HardwareProfile.model_validate(raw_profile)
    except ValidationError as exc:
        raise CompatibilityError(
            f"Invalid Hardware Profile {profile_path}: {exc}"
        ) from exc


class CompatibilityMatcher:
    """Evaluate hardware compatibility and immediate execution readiness."""

    def __init__(
        self,
        available_adapters: Collection[str] = (),
        model_file_exists: Callable[[ModelManifest], bool] | None = None,
    ) -> None:
        self.available_adapters = frozenset(available_adapters)
        self.model_file_exists = model_file_exists or (lambda manifest: False)

    def check(
        self,
        manifest: ModelManifest,
        hardware: HardwareProfile,
    ) -> CompatibilityReport:
        """Return deterministic blockers, warnings, and recommendations."""

        hardware_blockers: list[CompatibilityIssue] = []
        execution_blockers: list[CompatibilityIssue] = []
        warnings: list[CompatibilityIssue] = []
        recommendations: list[str] = []

        if manifest.riscv.require_rv64 and hardware.platform.architecture != "riscv64":
            hardware_blockers.append(
                CompatibilityIssue(
                    code="architecture_mismatch",
                    message=(
                        "Model requires RV64, current architecture is "
                        f"{hardware.platform.architecture}"
                    ),
                )
            )
            recommendations.append(
                "Run on a riscv64 target or provide a RISC-V Hardware Profile"
            )

        minimum_memory = manifest.resources.min_memory_mb
        if hardware.memory.total_mb < minimum_memory:
            hardware_blockers.append(
                CompatibilityIssue(
                    code="insufficient_total_memory",
                    message=(
                        f"Model requires at least {minimum_memory} MB of total memory, "
                        f"current host has {hardware.memory.total_mb} MB"
                    ),
                )
            )
            recommendations.append(
                f"Use a host with at least {minimum_memory} MB of total memory"
            )

        runtime_status = getattr(hardware.runtimes, manifest.runtime)
        if not runtime_status.available:
            execution_blockers.append(
                CompatibilityIssue(
                    code="runtime_unavailable",
                    message=f"Required runtime {manifest.runtime} was not detected",
                )
            )
            recommendations.append(self._runtime_recommendation(manifest.runtime))

        if manifest.format != "builtin" and not self.model_file_exists(manifest):
            article = "An" if manifest.format == "onnx" else "A"
            execution_blockers.append(
                CompatibilityIssue(
                    code="model_file_missing",
                    message=(
                        f"{article} {manifest.format.upper()} model file is required"
                    ),
                )
            )
            recommendations.append(
                f"Provide the required {manifest.format.upper()} model file"
            )

        if manifest.runtime not in self.available_adapters:
            execution_blockers.append(
                CompatibilityIssue(
                    code="adapter_unavailable",
                    message=(
                        f"No Runtime Adapter is available for {manifest.runtime}"
                    ),
                )
            )
            recommendations.append(
                f"Implement or install a {manifest.runtime} Runtime Adapter"
            )

        if manifest.riscv.prefer_rvv and hardware.riscv.is_riscv:
            if hardware.riscv.rvv is False:
                warnings.append(
                    CompatibilityIssue(
                        code="rvv_unavailable",
                        message="Model prefers RVV, but RVV was not detected",
                    )
                )
                recommendations.append(
                    "Use an RVV-capable target for better performance"
                )
            elif hardware.riscv.rvv is None:
                warnings.append(
                    CompatibilityIssue(
                        code="rvv_unknown",
                        message="RVV support could not be determined",
                    )
                )
                recommendations.append(
                    "Provide a Hardware Profile with explicit RVV information"
                )

        if (
            hardware.memory.total_mb >= minimum_memory
            and hardware.memory.available_mb < minimum_memory
        ):
            warnings.append(
                CompatibilityIssue(
                    code="low_available_memory",
                    message=(
                        f"Only {hardware.memory.available_mb} MB of memory is currently "
                        f"available; model recommends at least {minimum_memory} MB"
                    ),
                )
            )
            recommendations.append(
                f"Free at least {minimum_memory} MB of memory before running"
            )

        recommended_threads = manifest.resources.recommended_threads
        if (
            isinstance(recommended_threads, int)
            and recommended_threads > hardware.cpu.logical_cores
        ):
            warnings.append(
                CompatibilityIssue(
                    code="recommended_threads_exceed_cpu",
                    message=(
                        f"Model recommends {recommended_threads} threads, current CPU has "
                        f"{hardware.cpu.logical_cores} logical cores"
                    ),
                )
            )
            recommendations.append(
                f"Use at most {hardware.cpu.logical_cores} threads on this host"
            )

        blocking_reasons = [*hardware_blockers, *execution_blockers]
        return CompatibilityReport(
            model=manifest.name,
            compatible=not hardware_blockers,
            ready=not blocking_reasons,
            blocking_reasons=blocking_reasons,
            warnings=warnings,
            recommendations=list(dict.fromkeys(recommendations)),
        )

    @staticmethod
    def _runtime_recommendation(runtime: str) -> str:
        if runtime == "llama_cpp":
            return "Install llama.cpp and configure RVAI_LLAMA_CPP_BIN"
        if runtime == "onnxruntime":
            return "Install ONNX Runtime"
        return f"Install or enable the {runtime} runtime"
