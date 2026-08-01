"""Runtime availability checks used by hardware detection."""

from __future__ import annotations

import importlib.util
import os
import shutil
from collections.abc import Callable, Mapping

from rvai.hardware.schema import RuntimeInfo, RuntimeStatus


class RuntimeProbe:
    """Detect runtimes without importing or executing model backends."""

    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        which: Callable[[str], str | None] = shutil.which,
        find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    ) -> None:
        self.environ = os.environ if environ is None else environ
        self.which = which
        self.find_spec = find_spec

    def detect(self) -> RuntimeInfo:
        """Return the currently available runtime backends."""

        configured_llama = self.environ.get("RVAI_LLAMA_CPP_BIN")
        llama_executable = self.which(configured_llama or "llama-cli")
        onnx_available = self.find_spec("onnxruntime") is not None

        return RuntimeInfo(
            builtin=RuntimeStatus(available=True),
            llama_cpp=RuntimeStatus(
                available=llama_executable is not None,
                executable=llama_executable,
            ),
            onnxruntime=RuntimeStatus(available=onnx_available),
        )
