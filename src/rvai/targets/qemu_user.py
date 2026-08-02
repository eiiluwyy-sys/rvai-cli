"""QEMU user-mode target for riscv64 Linux workloads."""

from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Mapping
from pathlib import Path

from rvai.targets.base import ExecutionTarget, TargetError


def _normalize_architecture(architecture: str) -> str:
    normalized = architecture.lower()
    aliases = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "riscv64": "riscv64",
        "riscv32": "riscv32",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise TargetError(
            f"Unsupported QEMU host architecture: {architecture}"
        ) from exc


class QemuRiscv64Target(ExecutionTarget):
    """Launch the riscv64 benchmark through ``qemu-riscv64``."""

    def __init__(
        self,
        qemu_executable: Path | str | None = None,
        sysroot: Path | str | None = None,
        benchmark_executable: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        host_architecture: str | None = None,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._qemu_executable = qemu_executable
        self._sysroot = Path(sysroot) if sysroot is not None else None
        self._benchmark_executable = (
            Path(benchmark_executable)
            if benchmark_executable is not None
            else None
        )
        self._host_architecture = host_architecture

    @property
    def name(self) -> str:
        return "qemu-riscv64"

    @property
    def executable(self) -> Path:
        if self._benchmark_executable is not None:
            return self._benchmark_executable

        configured = self._environ.get("RVAI_RISCV64_BENCH_BIN")
        if configured:
            return Path(configured).expanduser()

        return (
            Path(__file__).resolve().parents[3]
            / "build-riscv64"
            / "rvai-bench"
        )

    @property
    def sysroot(self) -> Path:
        if self._sysroot is not None:
            return self._sysroot
        configured = self._environ.get("RVAI_RISCV64_SYSROOT")
        return (
            Path(configured).expanduser()
            if configured
            else Path("/usr/riscv64-linux-gnu")
        )

    def build_command(
        self,
        executable: Path,
        arguments: list[str],
    ) -> list[str]:
        qemu = self._resolve_qemu()
        if not self.sysroot.is_dir():
            raise TargetError(f"RISC-V sysroot not found: {self.sysroot}")
        if not executable.is_file():
            raise TargetError(f"RISC-V benchmark not found: {executable}")

        host_architecture = _normalize_architecture(
            self._host_architecture or platform.machine()
        )
        return [
            str(qemu),
            "-L",
            str(self.sysroot),
            "-E",
            "RVAI_EXECUTION_ENVIRONMENT=qemu-user",
            "-E",
            f"RVAI_HOST_ARCHITECTURE={host_architecture}",
            str(executable),
            *arguments,
        ]

    def _resolve_qemu(self) -> Path:
        configured = (
            self._qemu_executable
            or self._environ.get("RVAI_QEMU_RISCV64_BIN")
            or "qemu-riscv64"
        )
        candidate = Path(configured).expanduser()
        if candidate.parent != Path("."):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
            raise TargetError(
                "qemu-riscv64 was not found; set RVAI_QEMU_RISCV64_BIN"
            )

        discovered = shutil.which(str(configured))
        if discovered:
            return Path(discovered)
        raise TargetError(
            "qemu-riscv64 was not found; set RVAI_QEMU_RISCV64_BIN"
        )
