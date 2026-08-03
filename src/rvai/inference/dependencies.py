"""Lazy loading for optional ONNX inference dependencies."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

from rvai.inference.errors import InferenceDependencyError


@dataclass(frozen=True)
class OnnxDependencies:
    numpy: ModuleType
    onnxruntime: ModuleType
    pillow_image: ModuleType


def load_onnx_dependencies() -> OnnxDependencies:
    """Import optional dependencies only when ONNX inference is requested."""

    modules: dict[str, ModuleType] = {}
    missing: list[str] = []
    for distribution, module_name in (
        ("numpy", "numpy"),
        ("onnxruntime", "onnxruntime"),
        ("Pillow", "PIL.Image"),
    ):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            missing.append(distribution)
    if missing:
        names = ", ".join(missing)
        raise InferenceDependencyError(
            f"ONNX inference dependencies are missing ({names}); "
            'install them with pip install -e ".[onnx]"'
        )
    return OnnxDependencies(
        numpy=modules["numpy"],
        onnxruntime=modules["onnxruntime"],
        pillow_image=modules["PIL.Image"],
    )
