from pathlib import Path

import pytest

from rvai.hardware.linux import HardwareProbeError, LinuxSystemProbe


FIXTURES = Path(__file__).parent / "fixtures"


def make_probe(cpuinfo_name: str, meminfo_path: Path | None = None) -> LinuxSystemProbe:
    return LinuxSystemProbe(
        cpuinfo_path=FIXTURES / cpuinfo_name,
        meminfo_path=meminfo_path or FIXTURES / "meminfo.txt",
    )


def test_memory_info_converts_kilobytes_to_megabytes() -> None:
    memory = make_probe("cpuinfo-x86_64.txt").memory_info()

    assert memory.total_mb == 31944
    assert memory.available_mb == 21408


def test_non_riscv_architecture_does_not_claim_riscv_support() -> None:
    riscv = make_probe("cpuinfo-x86_64.txt").riscv_info("x86_64")

    assert riscv.is_riscv is False
    assert riscv.xlen is None
    assert riscv.rvv is None


def test_rv64gc_fixture_reports_xlen_and_no_rvv() -> None:
    riscv = make_probe("cpuinfo-riscv64-rv64gc.txt").riscv_info("riscv64")

    assert riscv.is_riscv is True
    assert riscv.xlen == 64
    assert riscv.isa == "rv64imafdc_zicsr_zifencei"
    assert riscv.extensions == ["i", "m", "a", "f", "d", "c", "zicsr", "zifencei"]
    assert riscv.rvv is False


def test_rvv_fixture_reports_vector_support() -> None:
    riscv = make_probe("cpuinfo-riscv64-rvv.txt").riscv_info("riscv64")

    assert riscv.rvv is True
    assert "v" in riscv.extensions
    assert "zve32x" in riscv.extensions


def test_missing_memavailable_is_a_project_error(tmp_path: Path) -> None:
    meminfo_path = tmp_path / "meminfo"
    meminfo_path.write_text("MemTotal: 1024 kB\n", encoding="utf-8")

    with pytest.raises(HardwareProbeError, match="Missing MemAvailable"):
        make_probe("cpuinfo-x86_64.txt", meminfo_path).memory_info()


def test_invalid_memory_field_is_a_project_error(tmp_path: Path) -> None:
    meminfo_path = tmp_path / "meminfo"
    meminfo_path.write_text(
        "MemTotal: unknown\nMemAvailable: 1024 kB\n", encoding="utf-8"
    )

    with pytest.raises(HardwareProbeError, match="Invalid MemTotal"):
        make_probe("cpuinfo-x86_64.txt", meminfo_path).memory_info()


def test_missing_system_file_is_a_project_error(tmp_path: Path) -> None:
    probe = LinuxSystemProbe(
        cpuinfo_path=FIXTURES / "cpuinfo-x86_64.txt",
        meminfo_path=tmp_path / "missing-meminfo",
    )

    with pytest.raises(HardwareProbeError, match="Cannot read system information"):
        probe.memory_info()
