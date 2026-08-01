"""Coordinator that builds the public hardware profile."""

from __future__ import annotations

import os
import platform

from rvai.hardware.linux import HardwareProbeError, LinuxSystemProbe
from rvai.hardware.runtime import RuntimeProbe
from rvai.hardware.schema import CpuInfo, HardwareProfile, PlatformInfo


_ARCHITECTURE_ALIASES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "riscv64": "riscv64",
    "riscv32": "riscv32",
}


def normalize_architecture(architecture: str) -> str:
    """Normalize only the architecture spellings supported by P1."""

    normalized = architecture.lower().strip()
    return _ARCHITECTURE_ALIASES.get(normalized, normalized or "unknown")


class HardwareProbe:
    """Collect Linux hardware data and runtime availability into one profile."""

    def __init__(
        self,
        system_probe: LinuxSystemProbe | None = None,
        runtime_probe: RuntimeProbe | None = None,
    ) -> None:
        self.system_probe = system_probe or LinuxSystemProbe()
        self.runtime_probe = runtime_probe or RuntimeProbe()

    def detect(self) -> HardwareProfile:
        """Detect the current host without starting any workload."""

        os_name = platform.system().lower()
        if os_name != "linux":
            raise HardwareProbeError(
                "Hardware detection currently supports Linux only, "
                f"not {os_name or 'unknown'}"
            )

        architecture = normalize_architecture(platform.machine())
        logical_cores = os.cpu_count() or 1
        return HardwareProfile(
            platform=PlatformInfo(
                os=os_name,
                kernel=platform.release() or "unknown",
                architecture=architecture,
            ),
            cpu=CpuInfo(logical_cores=logical_cores),
            memory=self.system_probe.memory_info(),
            riscv=self.system_probe.riscv_info(architecture),
            runtimes=self.runtime_probe.detect(),
        )
