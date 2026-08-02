from pathlib import Path

import pytest

from rvai.targets import NativeTarget, TargetError


def test_native_target_builds_direct_command(tmp_path: Path) -> None:
    executable = tmp_path / "rvai-bench"
    executable.touch(mode=0o755)
    target = NativeTarget(executable=executable)

    command = target.build_command(target.executable, ["gemm-int8", "--m", "8"])

    assert command == [str(executable), "gemm-int8", "--m", "8"]


def test_native_target_uses_environment_override(tmp_path: Path) -> None:
    executable = tmp_path / "custom-bench"
    executable.touch(mode=0o755)
    target = NativeTarget(environ={"RVAI_BENCH_BIN": str(executable)})

    assert target.build_command(target.executable, []) == [str(executable)]


def test_native_target_rejects_missing_benchmark(tmp_path: Path) -> None:
    target = NativeTarget(executable=tmp_path / "missing")

    with pytest.raises(TargetError, match="Native benchmark not found"):
        target.build_command(target.executable, [])
