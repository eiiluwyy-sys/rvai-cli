"""Native-host execution target."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from rvai.targets.base import ExecutionTarget, TargetError


class NativeTarget(ExecutionTarget):
    """Launch the host-native ``rvai-bench`` executable directly."""

    def __init__(
        self,
        executable: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._configured_executable = (
            Path(executable) if executable is not None else None
        )
        self._environ = os.environ if environ is None else environ

    @property
    def name(self) -> str:
        return "native"

    @property
    def executable(self) -> Path:
        if self._configured_executable is not None:
            return self._configured_executable

        configured = self._environ.get("RVAI_BENCH_BIN")
        if configured:
            return Path(configured).expanduser()

        discovered = shutil.which("rvai-bench")
        if discovered:
            return Path(discovered)

        return Path(__file__).resolve().parents[3] / "build" / "rvai-bench"

    def build_command(
        self,
        executable: Path,
        arguments: list[str],
    ) -> list[str]:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise TargetError(f"Native benchmark not found: {executable}")
        return [str(executable), *arguments]
