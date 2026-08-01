import json

from typer.testing import CliRunner

import rvai.cli as cli
from rvai.hardware import HardwareProbeError
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


def profile() -> HardwareProfile:
    return HardwareProfile(
        platform=PlatformInfo(os="linux", kernel="6.8.0", architecture="x86_64"),
        cpu=CpuInfo(logical_cores=16),
        memory=MemoryInfo(total_mb=31944, available_mb=21408),
        riscv=RiscVInfo(is_riscv=False),
        runtimes=RuntimeInfo(
            builtin=RuntimeStatus(available=True),
            llama_cpp=RuntimeStatus(available=False),
            onnxruntime=RuntimeStatus(available=False),
        ),
    )


def test_detect_outputs_stable_json(monkeypatch) -> None:
    class FakeHardwareProbe:
        def detect(self) -> HardwareProfile:
            return profile()

    monkeypatch.setattr(cli, "HardwareProbe", FakeHardwareProbe)

    result = runner.invoke(cli.app, ["detect"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == profile().model_dump()


def test_detect_hides_probe_tracebacks(monkeypatch) -> None:
    class FailingHardwareProbe:
        def detect(self) -> HardwareProfile:
            raise HardwareProbeError("Cannot read system information")

    monkeypatch.setattr(cli, "HardwareProbe", FailingHardwareProbe)

    result = runner.invoke(cli.app, ["detect"])

    assert result.exit_code == 1
    assert "Error: Cannot read system information" in result.output
    assert "Traceback" not in result.output
