"""Linux-specific sources for hardware detection."""

from __future__ import annotations

import re
from pathlib import Path

from rvai.hardware.schema import MemoryInfo, RiscVInfo


class HardwareProbeError(RuntimeError):
    """Raised when a hardware profile cannot be collected safely."""


class LinuxSystemProbe:
    """Read Linux system information from injectable procfs paths."""

    def __init__(
        self,
        cpuinfo_path: Path | str = Path("/proc/cpuinfo"),
        meminfo_path: Path | str = Path("/proc/meminfo"),
    ) -> None:
        self.cpuinfo_path = Path(cpuinfo_path)
        self.meminfo_path = Path(meminfo_path)

    def memory_info(self) -> MemoryInfo:
        """Read required memory fields from ``/proc/meminfo``."""

        values = self._read_key_values(self.meminfo_path)
        total_kb = self._parse_kilobytes(values, "MemTotal", self.meminfo_path)
        available_kb = self._parse_kilobytes(
            values, "MemAvailable", self.meminfo_path
        )
        return MemoryInfo(
            total_mb=total_kb // 1024,
            available_mb=available_kb // 1024,
        )

    def riscv_info(self, architecture: str) -> RiscVInfo:
        """Return conservative RISC-V ISA information for an architecture."""

        if architecture not in {"riscv32", "riscv64"}:
            return RiscVInfo(is_riscv=False)

        xlen = 64 if architecture == "riscv64" else 32
        values = self._read_key_values(self.cpuinfo_path)
        isa = values.get("isa")
        if isa is None:
            return RiscVInfo(is_riscv=True, xlen=xlen)

        extensions, rvv = self._parse_isa(isa)
        return RiscVInfo(
            is_riscv=True,
            xlen=xlen,
            isa=isa,
            extensions=extensions,
            rvv=rvv,
        )

    @staticmethod
    def _read_key_values(path: Path) -> dict[str, str]:
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError as exc:
            detail = exc.strerror or str(exc)
            raise HardwareProbeError(
                f"Cannot read system information from {path}: {detail}"
            ) from exc

        values: dict[str, str] = {}
        for line in contents.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                values.setdefault(key.strip().lower(), value.strip())
        return values

    @staticmethod
    def _parse_kilobytes(
        values: dict[str, str], field: str, source: Path
    ) -> int:
        raw_value = values.get(field.lower())
        if raw_value is None:
            raise HardwareProbeError(f"Missing {field} in {source}")

        match = re.fullmatch(r"(\d+)\s+kB", raw_value, flags=re.IGNORECASE)
        if match is None:
            raise HardwareProbeError(
                f"Invalid {field} value in {source}: {raw_value!r}"
            )
        return int(match.group(1))

    @staticmethod
    def _parse_isa(isa: str) -> tuple[list[str], bool | None]:
        normalized = isa.lower().strip()
        base, *named_extensions = normalized.split("_")
        match = re.fullmatch(r"rv(?:32|64)([a-z]+)", base)
        if match is None:
            return [item for item in named_extensions if item], None

        base_extensions = list(match.group(1))
        named_extensions = [item for item in named_extensions if item]
        extensions = [*base_extensions, *named_extensions]
        has_vector = (
            "v" in base_extensions
            or "v" in named_extensions
            or any(item.startswith("zve") for item in named_extensions)
        )
        return extensions, has_vector
