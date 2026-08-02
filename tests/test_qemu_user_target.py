from pathlib import Path

import pytest

import rvai.targets.qemu_user as qemu_user
from rvai.targets import QemuRiscv64Target, TargetError


def executable(path: Path) -> Path:
    path.touch(mode=0o755)
    return path


def configured_target(tmp_path: Path) -> QemuRiscv64Target:
    qemu = executable(tmp_path / "qemu-riscv64")
    sysroot = tmp_path / "sysroot"
    sysroot.mkdir()
    benchmark = executable(tmp_path / "rvai-bench")
    return QemuRiscv64Target(
        qemu_executable=qemu,
        sysroot=sysroot,
        benchmark_executable=benchmark,
        host_architecture="x86_64",
    )


def test_qemu_target_builds_complete_command(tmp_path: Path) -> None:
    target = configured_target(tmp_path)

    command = target.build_command(target.executable, ["gemm-int8", "--m", "32"])

    assert command == [
        str(tmp_path / "qemu-riscv64"),
        "-L",
        str(tmp_path / "sysroot"),
        "-E",
        "RVAI_EXECUTION_ENVIRONMENT=qemu-user",
        "-E",
        "RVAI_HOST_ARCHITECTURE=x86_64",
        str(tmp_path / "rvai-bench"),
        "gemm-int8",
        "--m",
        "32",
    ]


def test_qemu_target_uses_all_environment_overrides(tmp_path: Path) -> None:
    qemu = executable(tmp_path / "custom-qemu")
    sysroot = tmp_path / "custom-sysroot"
    sysroot.mkdir()
    benchmark = executable(tmp_path / "custom-bench")
    target = QemuRiscv64Target(
        environ={
            "RVAI_QEMU_RISCV64_BIN": str(qemu),
            "RVAI_RISCV64_SYSROOT": str(sysroot),
            "RVAI_RISCV64_BENCH_BIN": str(benchmark),
        },
        host_architecture="amd64",
    )

    command = target.build_command(target.executable, [])

    assert command[0] == str(qemu)
    assert command[2] == str(sysroot)
    assert command[-1] == str(benchmark)
    assert "RVAI_HOST_ARCHITECTURE=x86_64" in command


def test_qemu_target_rejects_missing_qemu(tmp_path: Path, monkeypatch) -> None:
    target = configured_target(tmp_path)
    target._qemu_executable = "qemu-riscv64"
    monkeypatch.setattr(qemu_user.shutil, "which", lambda name: None)

    with pytest.raises(TargetError, match="set RVAI_QEMU_RISCV64_BIN"):
        target.build_command(target.executable, [])


def test_qemu_target_rejects_missing_sysroot(tmp_path: Path) -> None:
    target = configured_target(tmp_path)
    target._sysroot = tmp_path / "missing-sysroot"

    with pytest.raises(TargetError, match="RISC-V sysroot not found"):
        target.build_command(target.executable, [])


def test_qemu_target_rejects_missing_benchmark(tmp_path: Path) -> None:
    target = configured_target(tmp_path)
    missing = tmp_path / "missing-bench"

    with pytest.raises(TargetError, match="RISC-V benchmark not found"):
        target.build_command(missing, [])
