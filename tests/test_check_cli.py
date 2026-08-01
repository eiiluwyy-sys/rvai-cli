import json
from pathlib import Path

from typer.testing import CliRunner

import rvai.cli as cli
from rvai.hardware.schema import (
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
    PlatformInfo,
    RiscVInfo,
    RuntimeInfo,
    RuntimeStatus,
)


runner = CliRunner()


def profile(architecture: str = "x86_64") -> HardwareProfile:
    is_riscv = architecture == "riscv64"
    return HardwareProfile(
        platform=PlatformInfo(
            os="linux", kernel="6.8.0", architecture=architecture
        ),
        cpu=CpuInfo(logical_cores=8),
        memory=MemoryInfo(total_mb=8192, available_mb=6144),
        riscv=RiscVInfo(
            is_riscv=is_riscv,
            xlen=64 if is_riscv else None,
            rvv=True if is_riscv else None,
        ),
        runtimes=RuntimeInfo(
            builtin=RuntimeStatus(available=True),
            llama_cpp=RuntimeStatus(available=True),
            onnxruntime=RuntimeStatus(available=True),
        ),
    )


def test_check_uses_live_hardware_by_default(monkeypatch) -> None:
    class FakeHardwareProbe:
        def detect(self) -> HardwareProfile:
            return profile()

    monkeypatch.setattr(cli, "HardwareProbe", FakeHardwareProbe)

    result = runner.invoke(cli.app, ["check", "qwen-small-int4"])

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["compatible"] is False
    assert "architecture_mismatch" in {
        issue["code"] for issue in report["blocking_reasons"]
    }


def test_check_reads_an_external_profile_without_probing(
    monkeypatch, tmp_path: Path
) -> None:
    class UnexpectedHardwareProbe:
        def detect(self) -> HardwareProfile:
            raise AssertionError("live probe should not run")

    monkeypatch.setattr(cli, "HardwareProbe", UnexpectedHardwareProbe)
    profile_path = tmp_path / "riscv64.json"
    profile_path.write_text(profile("riscv64").model_dump_json(), encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["check", "qwen-small-int4", "--profile", str(profile_path)],
    )

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["compatible"] is True
    assert "architecture_mismatch" not in {
        issue["code"] for issue in report["blocking_reasons"]
    }


def test_invalid_external_profile_is_a_user_facing_error(tmp_path: Path) -> None:
    profile_path = tmp_path / "invalid.json"
    profile_path.write_text("{invalid", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["check", "qwen-small-int4", "--profile", str(profile_path)],
    )

    assert result.exit_code == 1
    assert "Error: Invalid Hardware Profile JSON" in result.output
    assert "Traceback" not in result.output
